float32 x  # m, 同一帧内根据自车运动位置补偿后的x,对齐至最晚到达雷达时间
float32 y  # m, 同一帧内根据自车运动位置补偿后的y,对齐至最晚到达雷达时间
float32 z  # m ,预处理修正后的z
float32 r_xoy # m, 融合后的自车坐标系xoy平面内的range
float32 azimuth  # rad, 融合后的自车坐标系下的azimuth
float32 elevation  # rad, 融合后的自车坐标系下的elevation

float32 vr # m/s
float32 vr_azimuth # rad, 融合后的自车坐标系下的vr方向
float32 vr_compensation  # m/s 点云补偿速度
float32 vr2ground  # m/s, 补偿后的点云速度 vr2ground = vr + vr_compensation

uint8 motion_state  #点云动静态，根据vr2ground判断; 0静态，1动态

float32 power # dB
float32 snr # dB

uint8 radar_id #取 0,1,2,3,4,5
float64 timestamp # s