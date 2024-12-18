#include "utility.h"
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/Geoid.hpp>
#include <deque>
#include <mutex>

#define DEBUG_GPS_STATUS true

class GNSSOdom : public ParamServer
{
public:
  GNSSOdom(const rclcpp::NodeOptions &options) : ParamServer("simple_gps_odom", options)
  {
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "construct gnss odometry node!");
    subGPS = create_subscription<sensor_msgs::msg::NavSatFix>(gps_topic_name,
                                                              qos_gps, std::bind(&GNSSOdom::GNSSCB, this, std::placeholders::_1));
    subVcuOdom = create_subscription<nav_msgs::msg::Odometry>("/vcu_nav_odom", qos, 
      std::bind(&GNSSOdom::vcuOdomCb, this, std::placeholders::_1));
    
    left_odom_pub = create_publisher<nav_msgs::msg::Odometry>("/gps_odom", 100);
    init_origin_pub = create_publisher<nav_msgs::msg::Odometry>("/init_odom", 100);
    left_path_pub = create_publisher<nav_msgs::msg::Path>("/gps_path", 100);
    raw_gps_path_pub = create_publisher<nav_msgs::msg::Path>("/gps_raw_path", 100);

    //yaw_quat_left = geometry_msgs::msg::Quaternion();
    prev_pose_left = Eigen::Vector3d::Identity();
    // save_gps_tum_pose_raw_ofs.open("gps_tum_pose_raw.txt", std::ios::out | std::ios::app);
    // save_gps_tum_pose_ofs.open("gps_tum_pose.txt", std::ios::out | std::ios::app);
    //debug_angular_csv_ofs.open("debug_angular_ofs.csv", std::ios::out | std::ios::app);
    tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    curr_vcu_info = Eigen::Vector3d::Identity();

    debug_gps_status_ofs.open("debug_gps_status.csv");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPS;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subVcuOdom;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr left_odom_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr init_origin_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_path_pub;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr raw_gps_path_pub;

  std::shared_ptr<tf2_ros::Buffer> tfBuffer;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

  std::mutex mutexLock;
  bool initENU = false;
  nav_msgs::msg::Path left_path;
  GeographicLib::LocalCartesian geo_converter;
  Eigen::Vector3d prev_pose_left;
  geometry_msgs::msg::Quaternion yaw_quat_left;
  
  Eigen::Vector3d curr_vcu_info;

  std::mutex vcuMtx;

  bool isRotate = false;


  std::ofstream save_gps_tum_pose_ofs;

  std::ofstream save_gps_tum_pose_raw_ofs;

  std::ofstream debug_angular_csv_ofs;

  std::vector<double> angular_vec;
  std::ofstream debug_gps_status_ofs;

private:
  void GNSSCB(const sensor_msgs::msg::NavSatFix::SharedPtr msg) // note:这里不能加&，否则编译报错
  {
    
    uint8_t sat_num = msg->status.service >> 8;
    uint8_t gnss_status = msg->status.service % 256;
    uint8_t gnss_quality = msg->status.status;
    
    double horiPoseErr = msg->position_covariance[0];
    double vertPoseErr = msg->position_covariance[1];
    double hDop = msg->position_covariance[2];
    double vDop = msg->position_covariance[3];
    double satSnr = msg->position_covariance[4];

    if (DEBUG_GPS_STATUS)
    {
      debug_gps_status_ofs << static_cast<int>(gnss_status) << "," << static_cast<int>(gnss_quality) << "," << static_cast<int>(sat_num) 
                           << "," << horiPoseErr << "," << vertPoseErr << "," << hDop << "," << vDop 
                           << "," << satSnr <<  std::endl;
    }
    
    if (std::isnan(msg->latitude + msg->longitude + msg->altitude))
    {
      return;
    }
    
    Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->altitude);
    // std::cout << "LLA: " << lla.transpose() << std::endl;
    if (!initENU)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Init Orgin GPS LLA  %f, %f, %f", msg->latitude, msg->longitude, msg->altitude);
      // 根据状态判断
      // if (!isGpsValid(gnss_status, gnss_quality, sat_num, horiPoseErr))
      // {
      //   RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "gps status is not valid, reject this msg");
      //   return;
      // }
      geo_converter.Reset(lla[0], lla[1], lla[2]);
      initENU = true;
      /** publish initial pose from GNSS ENU Frame*/
      auto init_msg = nav_msgs::msg::Odometry();
      init_msg.header.stamp = msg->header.stamp;
      init_msg.header.frame_id = odometryFrame;
      init_msg.child_frame_id = "gps";
      init_msg.pose.pose.position.x = lla[0];
      init_msg.pose.pose.position.y = lla[1];
      init_msg.pose.pose.position.z = lla[2];
      init_msg.pose.covariance[0] = msg->position_covariance[0];
      init_msg.pose.covariance[7] = msg->position_covariance[4];
      init_msg.pose.covariance[14] = msg->position_covariance[8];
      init_msg.pose.pose.orientation = yaw_quat_left;
      init_origin_pub->publish(init_msg);
      return;
    }

    double x, y, z;
    // LLA->ENU, better accuacy than gpsTools especially for z value
    geo_converter.Forward(lla[0], lla[1], lla[2], x, y, z);
    Eigen::Vector3d enu(x, y, z);
    // if (abs(enu.x()) > 10000 || abs(enu.x()) > 10000 || abs(enu.x()) > 10000)
    // {
    //   /** check your lla coordinate */
    //   RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Error ogigin : %f, %f, %f", enu(0), enu(1), enu(2));
    //   return;
    // }
    bool orientationReady = false;
    double yaw = 0.0;
    double distance = sqrt(pow(enu(1) - prev_pose_left(1), 2) + pow(enu(0) - prev_pose_left(0), 2));
    // if (distance > 0.1)
    // {
    //   // 返回值是此点与原点连线与x轴正方向的夹角
    //   yaw = atan2(enu(1) - prev_pose_left(1), enu(0) - prev_pose_left(0));
    //   //Convert tf2::Quaternion to geometry_msgs::msg::Quaternion
    //   tf2::Quaternion tf2_quat;
    //   tf2_quat.setRPY(0.0, 0.0, yaw);
    //   yaw_quat_left = tf2::toMsg(tf2_quat);
    //   prev_pose_left = enu;
    //   orientationReady = true;
    // }

    yaw = atan2(enu(1), enu(0));
    // Convert tf2::Quaternion to geometry_msgs::msg::Quaternion
    tf2::Quaternion tf2_quat;
    tf2_quat.setRPY(0.0, 0.0, yaw);
    yaw_quat_left = tf2::toMsg(tf2_quat);
    
    
    tf2::Transform trans_gps;
    trans_gps.setOrigin(tf2::Vector3(enu(0), enu(1), enu(2)));
    tf2::Quaternion tf_gps;
    tf_gps.setRPY(0, 0, yaw);
    trans_gps.setRotation(tf_gps);

    tf2::Transform trans_gps_base;
    // trans_gps_base.setOrigin(tf2::Vector3(1.2599, -0.0619938, 1.43657));
    trans_gps_base.setOrigin(tf2::Vector3(0, 0, 0));
    tf2::Quaternion tf_gps_base;
    //tf_gps_base.setRPY(0, 0, 0); // 这里经过evo的工具对齐后，实际上的是131.5度，这里按135度来计算。
    tf_gps_base.setRPY(0, 0, 0); // 这里经过evo的工具对齐后，实际上的是131.5度，这里按135度来计算。
    //tf_gps_base.setRPY(0, 0, -131.5 / 180 * M_PI);
    trans_gps_base.setRotation(tf_gps_base);

    tf2::Transform trans_base = trans_gps_base * trans_gps;
    tf2::Transform trans_base_raw = trans_gps;
    
    //publish raw gps path
    static nav_msgs::msg::Path raw_gps_path;
    raw_gps_path.header.frame_id = odometryFrame;
    raw_gps_path.header.stamp = msg->header.stamp;

    auto p = geometry_msgs::msg::PoseStamped();
    p.header = raw_gps_path.header;
    p.pose.position.x = trans_base.getOrigin().x();
    p.pose.position.y = trans_base.getOrigin().y();
    p.pose.position.z = trans_base.getOrigin().z();
    p.pose.orientation.x = trans_base.getRotation().x();
    p.pose.orientation.y = trans_base.getRotation().y();
    p.pose.orientation.z = trans_base.getRotation().z();
    p.pose.orientation.w = trans_base.getRotation().w();
    raw_gps_path.poses.push_back(p);
    raw_gps_path_pub->publish(raw_gps_path);

    //1. 首先判断状态
    double currGpsTime = rclcpp::Time(msg->header.stamp).seconds();
    static double lastGpsTime = rclcpp::Time(msg->header.stamp).seconds();
    bool gpsStatusValid = isGpsValid(gnss_status, gnss_quality, sat_num, horiPoseErr);
    bool gpsPoseValid = isGpsPoseValid(prev_pose_left, enu, currGpsTime - lastGpsTime);

    if (!gpsStatusValid)
      return;
    
    if (!gpsPoseValid)
      return;
    // 在状态可用下，根据距离以及当前速度来判断，到车停下来时就不发。
    // if (distance < 0.1)
    // {
    //   RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "\033[33;1mgps dis %f <0.1m\033[0m]]", distance);
    //   return;
    // }
    
    lastGpsTime = currGpsTime;
    prev_pose_left = enu;
    geometry_msgs::msg::TransformStamped ts;
    ts.header.frame_id = odometryFrame;
    ts.child_frame_id = "/gps_link";
    ts.header.stamp = msg->header.stamp;
    // ts.transform.setOrigin(trans_base.getOrigin());
    // ts.transform.setRotation(trans_base.getRotation());
    ts.transform.translation.x = trans_base.getOrigin().x();
    ts.transform.translation.y = trans_base.getOrigin().y();
    ts.transform.translation.z = trans_base.getOrigin().z();
    ts.transform.rotation = tf2::toMsg(trans_base.getRotation());
    //tfBroadcaster->sendTransform(ts);

    /** pub gps odometry*/
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = rclcpp::Time((stamp2Sec(msg->header.stamp) + 0.3) * 1e9); // 1.0太大
    odom_msg.header.frame_id = odometryFrame;
    odom_msg.child_frame_id = "gps";

    odom_msg.pose.pose.position.x = trans_base.getOrigin().x();
    odom_msg.pose.pose.position.y = trans_base.getOrigin().y();
    odom_msg.pose.pose.position.z = trans_base.getOrigin().z();
    odom_msg.pose.covariance[0] = horiPoseErr;
    odom_msg.pose.covariance[7] = vertPoseErr;
    odom_msg.pose.covariance[14] = msg->position_covariance[8];
    odom_msg.pose.covariance[1] = lla[0];
    odom_msg.pose.covariance[2] = lla[1];
    odom_msg.pose.covariance[3] = lla[2];
    odom_msg.pose.covariance[4] = gnss_status;
    odom_msg.pose.covariance[5] = gnss_quality;
    odom_msg.pose.covariance[6] = sat_num;
    // odom_msg.pose.pose.orientation = yaw_quat_left;
    odom_msg.pose.pose.orientation.x = trans_base.getRotation().x();
    odom_msg.pose.pose.orientation.y = trans_base.getRotation().y();
    odom_msg.pose.pose.orientation.z = trans_base.getRotation().z();
    odom_msg.pose.pose.orientation.w = trans_base.getRotation().w();
    left_odom_pub->publish(odom_msg);

    /** just for gnss visualization */
    // publish path
    left_path.header.frame_id = odometryFrame;
    left_path.header.stamp = msg->header.stamp;

    auto pose = geometry_msgs::msg::PoseStamped();
    pose.header = left_path.header;
    pose.pose.position.x = trans_base.getOrigin().x();
    pose.pose.position.y = trans_base.getOrigin().y();
    pose.pose.position.z = trans_base.getOrigin().z();
    pose.pose.orientation.x = trans_base.getRotation().x();
    pose.pose.orientation.y = trans_base.getRotation().y();
    pose.pose.orientation.z = trans_base.getRotation().z();
    pose.pose.orientation.w = trans_base.getRotation().w();
    left_path.poses.push_back(pose);
    left_path_pub->publish(left_path);

    // save_gps_tum_pose_ofs << std::fixed << std::setprecision(8) <<
    //                       stamp2Sec(msg->header.stamp) << " " << trans_base.getOrigin().x() <<   " " << trans_base.getOrigin().y() << " " << 0.0
    //                       << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;

    // save_gps_tum_pose_raw_ofs << std::fixed << std::setprecision(8) <<
    //                       stamp2Sec(msg->header.stamp) << " " << trans_base_raw.getOrigin().x() <<   " " << trans_base_raw.getOrigin().y() << " " << 0.0
    //                       << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
  }

  bool isGpsValid(int gnss_status, int gnss_quality, int sat_num, double horiPose_Err)
  {
    // 检查状态是否可用，通过质量，状态，搜星个数来综合判断
    // 定位状态 0x5: Only Time Valid； 0x4: GNSS+DR; 0x3: 3D Fixed; 0x2: 2D Fixed; 0x1: DR; 0x0: Invalid
    //  cout << "quality:" << (size_t)msg->status.status << " status:" << msg->status.service%256
    //       << " sat_num:" << (size_t)(msg->status.service >> 8)  << endl;
    if ((gnss_status != 3) &&  (gnss_status != 4) && (gnss_status != 1)) //这里经过测试只出现这个状态，没有其他状态，理论上不只是这个状态可用，其他状态比如GNSS+DR，DR，
    {
      RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "gps status is %d not 3D Fixed ", gnss_status);
      return false;
    }
    // 0x6: Dead Reckoning; 0x5: RTK Float; 0x4: RTK Fixed; 0x2: DGPS; 0x1: Single Position; 0x0: Invalid
    if ((gnss_quality != 4) && (gnss_quality != 2) && (gnss_quality != 5))
    {
      RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "gps quality is %d  not RTK Fixed or  DGPS", gnss_quality);
      return false;
    }

    if (gnss_quality == 5) // 当为rtk float时， 且搜星个数大于10 且水平位置误差在0.1以下时
    {
      if (sat_num < 10 ||  horiPose_Err > 0.1)
        return false;
    }

    if (gnss_quality == 2 /*|| gnss_quality == 5*/) // 如果是DGPS,还要结合搜星个数来判断
    {
      if (sat_num < 1) // 10 
      {
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "gps quality is  DGPS, sat_num < 10, reject this msg");
        return false;
      }
    }


    return true;
  }

  bool isGpsPoseValid(Eigen::Vector3d& last_pose, Eigen::Vector3d& curr_pose, double dt)
  {
    //分别从横向，纵向，高度3个维度进行判断是否帧间发生比较大的跳跃
    //1. 横向的变化在1个车道； 根据角速率判断是否正在变道，如果不是在变道，但是横向有一个比较大的偏差，则说明gps有跳变
    //2. 纵向的变化根据车速来判断；
    //3. 高度的变化根据前后帧的变化量。所以需要把底盘的车速信息接进来
    double dx = fabs(curr_pose[0] - last_pose[0]);
    double dy = fabs(curr_pose[1] - last_pose[1]);
    //double dz = fabs(curr_pose[2] - last_pose[2]);
    //std::cout << dx << " " << dy << " " << dz << std::endl;
    double dis  = sqrt(dx * dx + dy * dy);

    double drive_dis = dt *  curr_vcu_info[0];

    if (dis > drive_dis + 1.0)
    {
      RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "gps dis %.4f > drive dis  %.4f", dis, drive_dis + 1.0);
      return false;
    }

    return true;
  }

  void vcuOdomCb(const nav_msgs::msg::Odometry::SharedPtr vcuMsg)
  {
    std::lock_guard<std::mutex> lock(vcuMtx);
    curr_vcu_info[0] = vcuMsg->twist.twist.linear.x;
    curr_vcu_info[1] = vcuMsg->twist.twist.angular.z;
  #if 1
    //这里的目的是大概计算出来转弯时以及的角速度变化率
    static double last_angular = -1.0;
    static bool startRotate = false;
    static bool endRotate = false;
    if (fabs(last_angular) > 0)
    {
      double delta_ang = fabs(curr_vcu_info[1] - last_angular);
      if (delta_ang > 0.5 && !startRotate)
      {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "start to rotate...");
        startRotate = true;
      }
      if (startRotate)
       angular_vec.push_back(curr_vcu_info[1]);
      
      if (delta_ang > 0.5 && startRotate)
      {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "end rotate...");
        endRotate = true;
        startRotate = false;
      }

      if (endRotate)
      {
        //开始统计角速率
        double angular_ave = std::accumulate(angular_vec.begin(), angular_vec.end(), 0.0);
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "average angular is %.4f", angular_ave * 180 / M_PI);
        endRotate = false;
        angular_vec.clear();
      }

      last_angular = curr_vcu_info[1];

    }
  #endif 
  #if 0
    //写到csv文件中，在表格中分析
    //定时写一个文件，10min 10 * 60 * 50 = 30000 / 10 = 3000
    static double last_time = stamp2Sec(vcuMsg->header.stamp)* 1e9;
    double curr_time = stamp2Sec(vcuMsg->header.stamp)* 1e9;
    static int file_cnt = 0;
    if (curr_time - last_time >=  5 * 60)
    {
      file_cnt++;
      last_time = curr_time;
      debug_angular_csv_ofs.close();
    }
      
    if (curr_time - last_time <  5 * 60)
    {
      std::string debug_angular_file = "debug_angular_file_" + std::to_string(file_cnt) + ".csv";
      static std::ofstream debug_angular_csv_ofs;
      
      if (!debug_angular_csv_ofs.is_open())
        debug_angular_csv_ofs.open(debug_angular_file, std::ios::app | std::ios::out);
      if (debug_angular_csv_ofs.is_open())
        debug_angular_csv_ofs << curr_vcu_info[0] << "," << curr_vcu_info[1] << std::endl;
    }
    
  #endif  
    //判断是否在变道或者转弯
    if (fabs(curr_vcu_info[1]) > 0.1) // 按照 0.1rad/s 0.1 * 180 / 3.1415 = 5.73°
    {
      isRotate = true;
    }


  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  // rclcpp::executors::MultiThreadedExecutor exec;
  auto simpleGpsOdom = std::make_shared<GNSSOdom>(options);
  // exec.add_node(simpleGpsOdom);

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Simple GPS Odmetry Started.\033[0m");
  // exec.spin();
  rclcpp::spin(simpleGpsOdom);
  rclcpp::shutdown();
  return 0;
}
