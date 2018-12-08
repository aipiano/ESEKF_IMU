import numpy as np
import numpy.linalg as la
import transformations as tr
import math


class ImuParameters:
    def __init__(self):
        self.frequency = 200
        self.sigma_a_n = 0.0     # acc noise.   m/(s*sqrt(s)), continuous noise sigma
        self.sigma_w_n = 0.0     # gyro noise.  rad/sqrt(s), continuous noise sigma
        self.sigma_a_b = 0.0     # acc bias     m/sqrt(s^5), continuous bias sigma
        self.sigma_w_b = 0.0     # gyro bias    rad/sqrt(s^3), continuous bias sigma


class ESEKF(object):
    def __init__(self, init_nominal_state: np.array, imu_parameters: ImuParameters):
        """
        :param init_nominal_state: [ p, q, v, a_b, w_b, g ], a 19x1 or 1x19 vector
        :param imu_parameters: imu parameters
        """
        self.nominal_state = init_nominal_state
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1    # force the quaternion has a positive real part.
        self.imu_parameters = imu_parameters

        # initialize noise covariance matrix
        noise_covar = np.zeros((12, 12))
        # assume the noises (especially sigma_a_n) are isotropic so that we can precompute self.noise_covar and save it.
        noise_covar[0:3, 0:3] = (imu_parameters.sigma_a_n**2) * np.eye(3)
        noise_covar[3:6, 3:6] = (imu_parameters.sigma_w_n**2) * np.eye(3)
        noise_covar[6:9, 6:9] = (imu_parameters.sigma_a_b**2) * np.eye(3)
        noise_covar[9:12, 9:12] = (imu_parameters.sigma_w_b**2) * np.eye(3)
        G = np.zeros((18, 12))
        G[3:6, 3:6] = -np.eye(3)
        G[6:9, 0:3] = -np.eye(3)
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)
        self.noise_covar = G @ noise_covar @ G.T

        # initialize error covariance matrix
        self.error_covar = 0.01 * self.noise_covar

        self.last_predict_time = 0.0

    def predict(self, imu_measurement: np.array):
        """
        :param imu_measurement: [t, w_m, a_m]
        :return: 
        """
        if self.last_predict_time == imu_measurement[0]:
            return
        # we predict error_covar first, because __predict_nominal_state will change the nominal state.
        self.__predict_error_covar(imu_measurement)
        self.__predict_nominal_state(imu_measurement)
        self.last_predict_time = imu_measurement[0]  # update timestamp

    def __update_legacy(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        An old implementation of the updating procedure.
        :param gt_measurement: [p, q], a 7x1 or 1x7 vector
        :param measurement_covar: a 7x7 symmetrical matrix
        :return: 
        """
        """
         Hx = dh/dx = [[I, 0, 0, 0, 0, 0]
                       [0, I, 0, 0, 0, 0]]
        """
        Hx = np.zeros((7, 19))
        Hx[0:3, 0:3] = np.eye(3)
        Hx[3:7, 3:7] = np.eye(4)

        """
         X = dx/d(delta_x) = [[I_3, 0, 0],
                              [0, Q_d_theta, 0],
                              [0, 0, I_12]
        """
        X = np.zeros((19, 18))
        q = self.nominal_state[3:7]
        X[0:3, 0:3] = np.eye(3)
        X[3:7, 3:6] = 0.5 * np.array([[-q[1], -q[2], -q[3]],
                                      [q[0], -q[3], q[2]],
                                      [q[3], q[0], -q[1]],
                                      [-q[2], q[1], q[0]]])
        X[7:19, 6:18] = np.eye(12)

        H = Hx @ X                      # 7x18
        PHt = self.error_covar @ H.T    # 18x7
        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        K = PHt @ la.inv(H @ PHt + measurement_covar)

        # update error covariance matrix
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        # force the error_covar to be a symmetrical matrix
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        """
        compute errors.
        restrict quaternions in measurement and state have positive real parts.
        this is necessary for errors computation since we subtract quaternions directly.
        """
        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        # NOTE: subtracting quaternion directly is tricky. that's why we abandon this implementation.
        errors = K @ (gt_measurement.reshape(-1, 1) - Hx @ self.nominal_state.reshape(-1, 1))

        # inject errors to the nominal state
        self.nominal_state[0:3] += errors[0:3, 0]  # update position
        dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
        # print(dq)
        self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # update rotation
        self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1
        self.nominal_state[7:] += errors[6:, 0]  # update the rest.

        """
        reset errors to zero and modify the error covariance matrix.
        we do nothing to the errors since we do not save them.
        but we need to modify the error_covar according to P = GPG^T
        """
        G = np.eye(18)
        G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
        self.error_covar = G @ self.error_covar @ G.T

    def update(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        :param gt_measurement: [p, q], a 7x1 or 1x7 vector
        :param measurement_covar: a 6x6 symmetrical matrix = diag{sigma_p^2, sigma_theta^2}
        :return: 
        """
        """
        we simulate a system that measure the errors between the nominal state and ground-truth state directly,
        so that we can avoid the direct subtracting of quaternions.
        
        we define q1 - q2 = conjugate(q2) x q1, so that q2 x (q1 - q2) = q1.
        
        ground_truth - nominal_state = delta = H @ error_state + noise
        """
        H = np.zeros((6, 18))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        PHt = self.error_covar @ H.T  # 18x6
        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6

        # update error covariance matrix
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        # force the error_covar to be a symmetrical matrix
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        # compute the measurements according to the nominal state and ground-truth state.
        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        gt_p = gt_measurement[0:3]
        gt_q = gt_measurement[3:7]
        q = self.nominal_state[3:7]

        delta = np.zeros((6, 1))
        delta[0:3, 0] = gt_p - self.nominal_state[0:3]
        delta_q = tr.quaternion_multiply(tr.quaternion_conjugate(q), gt_q)
        if delta_q[0] < 0:
            delta_q *= -1
        angle = math.asin(la.norm(delta_q[1:4]))
        if math.isclose(angle, 0):
            axis = np.zeros(3,)
        else:
            axis = delta_q[1:4] / la.norm(delta_q[1:4])
        delta[3:6, 0] = angle * axis

        # compute state errors.
        errors = K @ delta

        # inject errors to the nominal state
        self.nominal_state[0:3] += errors[0:3, 0]  # update position
        dq = tr.quaternion_about_axis(la.norm(errors[3:6, 0]), errors[3:6, 0])
        # print(dq)
        self.nominal_state[3:7] = tr.quaternion_multiply(q, dq)  # update rotation
        self.nominal_state[3:7] /= la.norm(self.nominal_state[3:7])
        if self.nominal_state[3] < 0:
            self.nominal_state[3:7] *= 1
        self.nominal_state[7:] += errors[6:, 0]  # update the rest.

        """
        reset errors to zero and modify the error covariance matrix.
        we do nothing to the errors since we do not save them.
        but we need to modify the error_covar according to P = GPG^T
        """
        G = np.eye(18)
        G[3:6, 3:6] = np.eye(3) - tr.skew_matrix(0.5 * errors[3:6, 0])
        self.error_covar = G @ self.error_covar @ G.T

    def __predict_nominal_state(self, imu_measurement: np.array):
        p = self.nominal_state[:3].reshape(-1, 1)
        q = self.nominal_state[3:7]
        v = self.nominal_state[7:10].reshape(-1, 1)
        a_b = self.nominal_state[10:13].reshape(-1, 1)
        w_b = self.nominal_state[13:16]
        g = self.nominal_state[16:19].reshape(-1, 1)

        w_m = imu_measurement[1:4].copy()
        a_m = imu_measurement[4:7].reshape(-1, 1).copy()
        dt = imu_measurement[0] - self.last_predict_time

        """
        dp/dt = v
        dv/dt = R(a_m - a_b) + g
        dq/dt = 0.5 * q x(quaternion product) (w_m - w_b)
        
        a_m and w_m are the measurements of IMU.
        a_b and w_b are biases of acc and gyro, respectively.
        R = R{q}, which bring the point from local coordinate to global coordinate.
        """
        w_m -= w_b
        a_m -= a_b

        # use the zero-order integration to integrate the quaternion.
        # q_{n+1} = q_n x q{(w_m - w_b) * dt}
        angle = la.norm(w_m)
        axis = w_m / angle
        R_w = tr.rotation_matrix(0.5 * dt * angle, axis)
        q_w = tr.quaternion_from_matrix(R_w, True)
        q_half_next = tr.quaternion_multiply(q, q_w)

        R_w = tr.rotation_matrix(dt * angle, axis)
        q_w = tr.quaternion_from_matrix(R_w, True)
        q_next = tr.quaternion_multiply(q, q_w)
        if q_next[0] < 0:   # force the quaternion has a positive real part.
            q_next *= -1

        # use RK4 method to integration velocity and position.
        # integrate velocity first.
        R = tr.quaternion_matrix(q)[:3, :3]
        R_half_next = tr.quaternion_matrix(q_half_next)[:3, :3]
        R_next = tr.quaternion_matrix(q_next)[:3, :3]
        v_k1 = R @ a_m + g
        v_k2 = R_half_next @ a_m + g
        # v_k3 = R_half_next @ a_m + g  # yes. v_k2 = v_k3.
        v_k3 = v_k2
        v_k4 = R_next @ a_m + g
        v_next = v + dt * (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4) / 6

        # integrate position
        p_k1 = v
        p_k2 = v + 0.5 * dt * v_k1  # k2 = v_next_half = v + 0.5 * dt * v' = v + 0.5 * dt * v_k1(evaluate at t0)
        p_k3 = v + 0.5 * dt * v_k2  # v_k2 is evaluated at t0 + 0.5*delta
        p_k4 = v + dt * v_k3
        p_next = p + dt * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4) / 6

        self.nominal_state[:3] = p_next.reshape(3,)
        self.nominal_state[3:7] = q_next
        self.nominal_state[7:10] = v_next.reshape(3,)
        # print(q_next)

    def __predict_error_covar(self, imu_measurement: np.array):
        w_m = imu_measurement[1:4]
        a_m = imu_measurement[4:7]
        a_b = self.nominal_state[9:12]
        w_b = self.nominal_state[12:15]
        q = self.nominal_state[3:7]
        R = tr.quaternion_matrix(q)[:3, :3]

        F = np.zeros((18, 18))
        F[0:3, 6:9] = np.eye(3)
        F[3:6, 3:6] = -tr.skew_matrix(w_m - w_b)
        F[3:6, 12:15] = -np.eye(3)
        F[6:9, 3:6] = -R @ tr.skew_matrix(a_m - a_b)
        F[6:9, 9:12] = -R

        # use 3rd-order truncated integration to compute transition matrix Phi.
        dt = imu_measurement[0] - self.last_predict_time
        Fdt = F * dt
        Fdt2 = Fdt @ Fdt
        Fdt3 = Fdt2 @ Fdt
        Phi = np.eye(18) + Fdt + 0.5 * Fdt2 + (1. / 6.) * Fdt3

        """
        use trapezoidal integration to integrate noise covariance:
          Qd = 0.5 * dt * (Phi @ self.noise_covar @ Phi.T + self.noise_covar)
          self.error_covar = Phi @ self.error_covar @ Phi.T + Qd
          
        operations above can be merged to the below for efficiency.
        """
        Qc_dt = 0.5*dt*self.noise_covar
        self.error_covar = Phi @ (self.error_covar + Qc_dt) @ Phi.T + Qc_dt

