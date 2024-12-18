int8 valid
float64 time_stamp #当前系统时间戳
float32 odo_pose_x      #x方向上的位置
float32 odo_pose_y      #y方向上的位置
float32 odo_pose_z
float32 odo_pose_delta_x
float32 odo_pose_delta_y
float32 odo_pose_delta_z

float32 odo_yaw 
float32 odo_pitch
float32 odo_roll
float32 odo_delta_yaw #相邻时间间隔的yaw变化量
float32 odo_delta_pitch
float32 odo_delta_roll


float32 wheel_odo_speed
float32 wheel_odo_yawrate

# gps + imu
float32 twist_linear_x #x方向上线速度(车速)
float32 twist_linear_y
float32 twist_linear_z
float32 twist_linear_acc_x
float32 twist_linear_acc_y
float32 twist_linear_acc_z
float32 twist_angular_x
float32 twist_angular_y
float32 twist_angular_z #在xoy平面上的角速度(yaw_rate)


float64 latitude
float64 longitude
float64 altitude
float32 gps_yaw
float32 gps_pitch 
float32 gps_roll
int8 gps_type
