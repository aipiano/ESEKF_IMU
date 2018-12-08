import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import yaml
import math
from esekf import *


def load_imu_parameters():
    f = open('./data/params.yaml', 'r')
    yml = yaml.load(f.read())
    params = ImuParameters()
    params.frequency = yml['IMU.frequency']
    params.sigma_a_n = yml['IMU.acc_noise_sigma']  # m/sqrt(s^3)
    params.sigma_w_n = yml['IMU.gyro_noise_sigma']  # rad/sqrt(s)
    params.sigma_a_b = yml['IMU.acc_bias_sigma']     # m/sqrt(s^5)
    params.sigma_w_b = yml['IMU.gyro_bias_sigma']    # rad/sqrt(s^3)
    f.close()
    return params


def main():
    imu_data = np.loadtxt('./data/imu_noise.txt')
    gt_data = np.loadtxt('./data/traj_gt.txt')

    imu_parameters = load_imu_parameters()

    init_nominal_state = np.zeros((19,))
    init_nominal_state[:10] = gt_data[0, 1:]                # init p, q, v
    init_nominal_state[10:13] = 0                           # init ba
    init_nominal_state[13:16] = 0                           # init bg
    init_nominal_state[16:19] = np.array([0, 0, -9.81])     # init g
    estimator = ESEKF(init_nominal_state, imu_parameters)

    test_duration_s = [0., 61.]
    start_time = imu_data[0, 0]
    mask_imu = np.logical_and(imu_data[:, 0] <= start_time + test_duration_s[1],
                              imu_data[:, 0] >= start_time + test_duration_s[0])
    mask_gt = np.logical_and(gt_data[:, 0] <= start_time + test_duration_s[1],
                             gt_data[:, 0] >= start_time + test_duration_s[0])

    imu_data = imu_data[mask_imu, :]
    gt_data = gt_data[mask_gt, :]

    traj_est = [gt_data[0, :8]]
    update_ratio = 10    # control the frequency of ekf updating.
    sigma_measurement_p = 0.02   # in meters
    sigma_measurement_q = 0.015  # in rad
    sigma_measurement = np.eye(6)
    sigma_measurement[0:3, 0:3] *= sigma_measurement_p**2
    sigma_measurement[3:6, 3:6] *= sigma_measurement_q**2
    for i in range(1, imu_data.shape[0]):
        timestamp = imu_data[i, 0]
        estimator.predict(imu_data[i, :])
        if i % update_ratio == 0:
            # we assume the timestamps are aligned.
            assert math.isclose(gt_data[i, 0], timestamp)
            gt_pose = gt_data[i, 1:8].copy()  # gt_pose = [p, q]
            # add position noise
            gt_pose[:3] += np.random.randn(3,) * sigma_measurement_p
            # add rotation noise, u = [1, 0.5 * noise_angle_axis]
            # u = 0.5 * np.random.randn(4,) * sigma_measurement_q
            # u[0] = 1
            u = np.random.randn(3, ) * sigma_measurement_q
            qn = tr.quaternion_about_axis(la.norm(u), u / la.norm(u))
            gt_pose[3:] = tr.quaternion_multiply(gt_pose[3:], qn)
            # update filter by measurement.
            estimator.update(gt_pose, sigma_measurement)

        print('[%f]:' % timestamp, estimator.nominal_state)
        frame_pose = np.zeros(8,)
        frame_pose[0] = timestamp
        frame_pose[1:] = estimator.nominal_state[:7]
        traj_est.append(frame_pose)

    # save trajectory to TUM format
    traj_est = np.array(traj_est)
    np.savetxt('./data/traj_gt_out.txt', gt_data[:, :8])
    np.savetxt('./data/traj_esekf_out.txt', traj_est)


if __name__ == '__main__':
    main()
