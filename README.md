# ESEKF_IMU
A python implemented error-state extended Kalman Filter. Suit for learning EKF and IMU integration.

Use simulated imu data (./data/imu_noise.txt) as input.

Output an trajectory estimated by esekf (./data/traj_esekf_out.txt) and a ground truth trajectory (./data/traj_gt_out.txt).

You can use [evo](https://github.com/MichaelGrupp/evo) to show both trajectories above.
> evo_traj tum ./data/traj_esekf_out.txt --ref ./data/traj_gt_out.txt -p

!(trajectory)[https://github.com/aipiano/ESEKF_IMU/blob/master/images/trajectory.png?raw=true]

!(xyz)[https://github.com/aipiano/ESEKF_IMU/blob/master/images/xyz.png?raw=true]

!(rpy)[https://github.com/aipiano/ESEKF_IMU/blob/master/images/rpy.png?raw=true]
