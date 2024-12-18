#include "utility.h"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include "gpal_msgs/msg/vcu_data.hpp"

// 以下的pointType不是点云的类型 是位姿Pose的类型
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

// 6自由度的位姿和其对应的时间
typedef PointXYZIRPYT PointTypePose;

class PointsFusion : public ParamServer
{

public:

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRadarCloudSurround;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointCloud;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subPose;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr aloopIsClosed_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr finish_mapping_sub;
  double timeRadarInfoCur;
  rclcpp::Time timeRadarInfoStamp;
  float transformTobeMapped[6];
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  nav_msgs::msg::Path local_pose_;

  pcl::PointCloud<PointType>::Ptr radarCloudRaw;
  pcl::PointCloud<PointType>::Ptr pointCloudAll;
  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  vector<pcl::PointCloud<PointType>::Ptr> allCloudKeyFrames;
  std::vector<double> keyFramePoseTimestamp;
  // 当前点云帧在path的哪两个id中 与关键帧点云与位姿一同push
  std::vector<int> nex_ind;
  std::vector<int> bef_ind;
  std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
  std::deque<double> timeStampQueue;
  sensor_msgs::msg::PointCloud2 currentCloudMsg;
  double point_time;
  std::mutex poseMtx;
  std::mutex pointMtx;
  bool aloop_is_closed_;
  int aloop_times = 0;
  int start_index = 0;
  PointsFusion(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_pointsFusion", options)
  {

    pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/local_map", 1);
    pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/cloud_registered", 1);
    pubRadarCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/map_global", 1);
    pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/trajectory", 1);

    subPointCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointCloudTopic, qos_lidar,
        std::bind(&PointsFusion::pointCloudInfoHandler, this, std::placeholders::_1));      

    subPose = create_subscription<nav_msgs::msg::Path>(
        "opt_path", qos_imu,
        std::bind(&PointsFusion::poseInfoHandler, this, std::placeholders::_1));

    aloopIsClosed_info_sub_ = create_subscription<std_msgs::msg::Bool>("updatePose", qos, std::bind(&PointsFusion::aloopIsClosedSubCb, this, std::placeholders::_1));
    
    // 结束建图 保存轨迹+gp
    finish_mapping_sub = create_subscription<std_msgs::msg::Empty>(
        "finish_map", 1, std::bind(&PointsFusion::finishMappingSub, this, std::placeholders::_1));

    allocateMemory();
  }

  void allocateMemory()
  {

    radarCloudRaw.reset(new pcl::PointCloud<PointType>()); 
    pointCloudAll.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

  }

  void aloopIsClosedSubCb(const std_msgs::msg::Bool &loopclosed_msgs)
  {
    aloop_is_closed_ = loopclosed_msgs.data;
    if (aloop_is_closed_) // 闭环更新
    {
      aloop_times++;
    }

    if (aloop_times == 2)
    {
      aloop_times = 0;
      aloop_is_closed_ = true;
      RCLCPP_WARN(get_logger(), "update slot pose!");
    }
    else
    {
      aloop_is_closed_ = false;
    }
  }

  void pointCloudInfoHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msgIn)
  {
    std::lock_guard<std::mutex> lock(pointMtx);
    // std::cout << "pointCloudInfoHandler" << std::endl;
    // 点云预处理 点云存入cloudQueue
    // pointCloudPreprocessing(msgIn);
    cloudQueue.push_back(*msgIn);
    
  }

  void pointCloudPreprocessing(const sensor_msgs::msg::PointCloud2::SharedPtr &radarCloudMsg)
  {

    pcl::fromROSMsg(*radarCloudMsg, *radarCloudRaw);
    if(!radarCloudRaw->points.empty())
    {
      // 点云预处理
      pcl::RadiusOutlierRemoval<PointType> outstrem;
      outstrem.setInputCloud(radarCloudRaw);
      outstrem.setRadiusSearch(0.5); // 1m范围内至少有2个点
      outstrem.setMinNeighborsInRadius(3);
      // outstrem.setKeepOrganized(true);
      outstrem.filter(*radarCloudRaw);
      // std::cout << "after radius removal:" << laserCloudIn->points.size() << std::endl;
      // 将x轴前2m, 后1m, y轴 左右1m, z轴1m内的点滤除
      pcl::PassThrough<PointType> pass;
      pass.setInputCloud(radarCloudRaw);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(-1.0, 2.0);
      pass.setFilterLimitsNegative(true);
      pass.filter(*radarCloudRaw);
      // //std::cout << "after pass through x:" << laserCloudIn->points.size() << std::endl;
      pass.setFilterFieldName("y");
      pass.setFilterLimits(-1.0, 1.0);
      pass.setFilterLimitsNegative(true);
      pass.filter(*radarCloudRaw);
      // std::cout << "after pass through y:" << laserCloudIn->points.size() << std::endl;

      pass.setFilterFieldName("z");
      pass.setFilterLimits(filter_min_z, 50.0); // 10.0
      pass.setFilterLimitsNegative(false);
      pass.filter(*radarCloudRaw);
      // std::cout << "after pass through:" << laserCloudIn->points.size() << std::endl;
      // 移除离群点滤波
      pcl::StatisticalOutlierRemoval<PointType> sor;
      sor.setInputCloud(radarCloudRaw);
      sor.setMeanK(50);
      sor.setStddevMulThresh(1);
      sor.filter(*radarCloudRaw);
      // 存入队列
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(*radarCloudRaw, output);
    }

  }

  void cloudQueueAlign()
  {
    // 先判断队列是否为空
    if(cloudQueue.empty() || local_pose_.poses.size()==0)
    {
      return;
    }

    // 时间戳判断
    pcl::PointCloud<PointType>::Ptr radarCloudIn(new pcl::PointCloud<PointType>());

    currentCloudMsg = std::move(cloudQueue.front());     // 拿到当前点云
    cloudQueue.pop_front();                              // 队列里pop一个
    pcl::moveFromROSMsg(currentCloudMsg, *radarCloudIn); // 转成ros消息 laserCloudIn
    point_time = stamp2Sec(currentCloudMsg.header.stamp);

    // 判断雷达数据的时间是否在位姿队列范围内
    while(!cloudQueue.empty())
    {
      if(point_time < stamp2Sec(local_pose_.poses[0].header.stamp))
      {
        // std::cout << "point_time < stamp2Sec(local_pose_.poses[0].header.stamp)" << std::endl;
        return;
      }
      else if(point_time > stamp2Sec(local_pose_.poses[local_pose_.poses.size() - 1].header.stamp))
      {
        // std::cout << "sleep 200ms" << std::endl;
        rclcpp::Time start = rclcpp::Clock().now();
        rclcpp::sleep_for(std::chrono::microseconds(2000));
      }
      else
      {
        break;
      }
    }

    if(cloudQueue.empty() || local_pose_.poses.size()==0)
    {
      return;
    }

    // std::cout << "cloudQueue.size(): " << cloudQueue.size() << std::endl;
    RCLCPP_ERROR(get_logger(), " point_time:%.8f",point_time);
    PointTypePose pointPose;
    if(pointPoseInterpolation(point_time, pointPose))
    {
      // 将点云、其对应的位姿、id存入队列
      allCloudKeyFrames.push_back(radarCloudIn);
      PointType thisPose3D;
      thisPose3D.x = pointPose.x;
      thisPose3D.y = pointPose.y;
      thisPose3D.z = pointPose.z;
      thisPose3D.intensity = cloudKeyPoses3D->size();
      cloudKeyPoses3D->push_back(thisPose3D);

      PointTypePose thisPose6D;
      thisPose6D.x = thisPose3D.x;
      thisPose6D.y = thisPose3D.y;
      thisPose6D.z = thisPose3D.z;
      thisPose6D.intensity = thisPose3D.intensity;
      thisPose6D.roll = pointPose.roll;
      thisPose6D.pitch = pointPose.pitch;
      thisPose6D.yaw = pointPose.yaw;
      thisPose6D.time = point_time;
      cloudKeyPoses6D->push_back(thisPose6D);
      keyFramePoseTimestamp.push_back(point_time);
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "pose interpolation failure!");
    }

  }

  bool pointPoseInterpolation(double time, PointTypePose keyPose)
  {
    nav_msgs::msg::Path curr_local_pose = local_pose_;
    if (curr_local_pose.poses.size() == 0)
    {
      RCLCPP_ERROR(get_logger(), "slotPoseUpdate local pose is wrong!");
      return false;
    }

    double newest_t = stamp2Sec(curr_local_pose.poses[curr_local_pose.poses.size() - 1].header.stamp);
    double oldest_t = stamp2Sec(curr_local_pose.poses[0].header.stamp);
    // std::cout << "time - newest_t: " << time - newest_t << std::endl;
    // std::cout << "time - oldest_t: " << time - oldest_t << std::endl;
    if(time - newest_t > 1.0 || time < oldest_t)
    {
      RCLCPP_ERROR(get_logger(), "slotPoseUpdate time is out of range!");
      return false;        
    }   

    std::vector<double> path_times;

// #pragma omp parallel for num_threads(numberOfCores)   
    for (int i = 0; i < curr_local_pose.poses.size(); ++i)
    {
      // std::cout << "i: " << i << std::endl;
      path_times.push_back(stamp2Sec(curr_local_pose.poses[i].header.stamp)); 
    }
    // std::cout << "path_times.size(): " << path_times.size() << std::endl;


    // 找到离time最近的那个时刻
    // std::cout << "time: " << time << std::endl;
    int after_ind = linearSearchClosestTime(path_times, time);
    // std::cout << "===>>after_ind: " << after_ind << std::endl;
    if (after_ind == -1 || after_ind == 0)
    {
      RCLCPP_ERROR(get_logger(), "linearSearchClostTime failure!");
      return false;
    }

    int before_ind = after_ind - 1;
    // std::cout << "after_ind: " << after_ind << " before_ind: " << before_ind << std::endl;
    double curr_local_time = stamp2Sec(curr_local_pose.poses[after_ind].header.stamp);
    double last_local_time = stamp2Sec(curr_local_pose.poses[before_ind].header.stamp);

    RCLCPP_DEBUG(get_logger(), "curr_local_time:%.8f last_local_time:%.8f", curr_local_time, last_local_time);

    double curr_local_pose_x = curr_local_pose.poses[after_ind].pose.position.x;
    double curr_local_pose_y = curr_local_pose.poses[after_ind].pose.position.y;

    double last_local_pose_x = curr_local_pose.poses[before_ind].pose.position.x;
    double last_local_pose_y = curr_local_pose.poses[before_ind].pose.position.y;

    double curr_yaw, last_yaw, curr_roll, curr_pitch, last_roll, last_pitch;
    tf2::Quaternion quat;
    tf2::fromMsg(curr_local_pose.poses[after_ind].pose.orientation, quat);
    tf2::Matrix3x3(quat).getRPY(curr_roll, curr_pitch, curr_yaw);
    tf2::fromMsg(curr_local_pose.poses[before_ind].pose.orientation, quat);
    tf2::Matrix3x3(quat).getRPY(last_roll, last_pitch, last_yaw);

    RCLCPP_DEBUG(get_logger(), "curr_yaw:%f, last_yaw:%f",curr_yaw, last_yaw);

    double dt = curr_local_time - last_local_time;

    double v = sqrt((last_local_pose_x - curr_local_pose_x) * (last_local_pose_x - curr_local_pose_x) +
                    (last_local_pose_y - curr_local_pose_y) * (last_local_pose_y - curr_local_pose_y)) / dt;

    // 经调试，这里的角度差存在判断错误的情况
    double delta_angle = angle_diff(curr_yaw, last_yaw);

    double w = delta_angle / dt;

    //std::cout << "v, w, dt:" << v << " " << w << " " << dt << std::endl;
    // 23.7.24 change to rosdebug
    RCLCPP_DEBUG(get_logger(), "v, w, dt:%f %f %f",v, w, dt);

    double delta_t = time - last_local_time;

    keyPose.yaw = last_yaw + delta_t * w;
    if (keyPose.yaw > M_PI)
      keyPose.yaw -= 2 * M_PI;
    else if (keyPose.yaw < -M_PI)
      keyPose.yaw += 2 * M_PI;

    keyPose.roll = curr_roll;
    keyPose.pitch = curr_pitch;
    //std::cout << "delta_t:" << delta_t << " interpolation yaw:" << keyPose.yaw << std::endl;
    // 23.7.24 change to rosdebug
    RCLCPP_DEBUG(get_logger(), "delta_t:%f interpolation yaw:%f",delta_t,  keyPose.yaw);

    double dx = v * cos(keyPose.yaw) * delta_t;
    double dy = v * sin(keyPose.yaw) * delta_t;

    keyPose.x = last_local_pose_x + dx;
    keyPose.y = last_local_pose_y + dy;
    keyPose.z = curr_local_pose.poses[before_ind].pose.position.z;

    // 记录id
    bef_ind.push_back(before_ind);
    nex_ind.push_back(after_ind);
    return true;

  }

  double angle_diff(double a, double b)
  {
    double d1, d2;
    d1 = a - b;
    d2 = 2 * M_PI - fabs(d1);
    if (d1 > 0)
      d2 *= -1.0;
    if (fabs(d1) < fabs(d2))
      return d1;
    else
      return d2;
  }

  int linearSearchClosestTime(std::vector<double> all_times, double time)
  {
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = start_index; i < all_times.size(); ++i)
    {
      // 从头开始，找到第一个比time大的时间 因为这个是有序排列的
      if (fabs(all_times[i] - time) < 0.5)
      {
        // 下次就从上一次的起点开始找
        start_index = i;
        return i;
      }
    }
    return -1;
  }

  void poseInfoHandler(const nav_msgs::msg::Path::SharedPtr msgIn)
  {
    std::lock_guard<std::mutex> lock(poseMtx);
    // std::cout << "poseInfoHandler" << std::endl;
    local_pose_ = *msgIn;
    // updatePose();
  }

  void updatePose()
  {
    // 对所有点云进行更新
// #pragma omp parallel for num_threads(numberOfCores)
    for(int i = 0; i < nex_ind.size(); ++i)
    {
      if(!updateRadarPose(i))
      {
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;31m----> point cloud update failure!.\033[0m");
      }
    }
    publishGlobalMap();
  }

  bool updateRadarPose(int ind)
  {
    if(local_pose_.poses.size() == 0)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;31m----> local_pose_.poses.size() == 0!.\033[0m");
      return false;
    }
    // 获取对应顺序的点云、位姿以及其在path中的id
    pcl::PointCloud<PointType>::Ptr radarPointCloud(new pcl::PointCloud<PointType>());
    double curRadarTime;
    int bef_id, next_id;
    // std::cout << allCloudKeyFrames.size() << " " << keyFramePoseTimestamp.size() << " " << bef_ind.size() << " " << nex_ind.size() << std::endl;
    radarPointCloud = allCloudKeyFrames[ind];
    curRadarTime = keyFramePoseTimestamp[ind];
    bef_id = bef_ind[ind];
    next_id = nex_ind[ind];

    // std::cout << "bef_id: " << bef_id << " next_id: " << next_id << std::endl;

    // 获取path开始与末尾的时间戳
    // std::cout << "local_pose_.poses.size(): " << local_pose_.poses.size() << std::endl;
    double newest_t = stamp2Sec(local_pose_.poses[local_pose_.poses.size() - 1].header.stamp);
    double oldest_t = stamp2Sec(local_pose_.poses[0].header.stamp);
    // RCLCPP_INFO(get_logger(), "newest_t:%.8f oldest_t:%.8f", newest_t, oldest_t);
    if(curRadarTime < oldest_t || curRadarTime > newest_t)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;31m----> point cloud time is wrong!.\033[0m");
      return false;
    }

    double curr_local_time = stamp2Sec(local_pose_.poses[next_id].header.stamp);
    double last_local_time = stamp2Sec(local_pose_.poses[bef_id].header.stamp);

    // RCLCPP_INFO(get_logger(), "curr_local_time:%.8f last_local_time:%.8f", curr_local_time, last_local_time);

    double curr_local_pose_x = local_pose_.poses[next_id].pose.position.x;
    double curr_local_pose_y = local_pose_.poses[next_id].pose.position.y;

    double last_local_pose_x = local_pose_.poses[bef_id].pose.position.x;
    double last_local_pose_y = local_pose_.poses[bef_id].pose.position.y;

    double curr_yaw, last_yaw, curr_roll, curr_pitch, last_roll, last_pitch;
    tf2::Quaternion quat;
    tf2::fromMsg(local_pose_.poses[next_id].pose.orientation, quat);
    tf2::Matrix3x3(quat).getRPY(curr_roll, curr_pitch, curr_yaw);
    tf2::fromMsg(local_pose_.poses[bef_id].pose.orientation, quat);
    tf2::Matrix3x3(quat).getRPY(last_roll, last_pitch, last_yaw);

    RCLCPP_DEBUG(get_logger(), "curr_yaw:%f, last_yaw:%f",curr_yaw, last_yaw);

    double dt = curr_local_time - last_local_time;

    double v = sqrt((last_local_pose_x - curr_local_pose_x) * (last_local_pose_x - curr_local_pose_x) +
                    (last_local_pose_y - curr_local_pose_y) * (last_local_pose_y - curr_local_pose_y)) / dt;

    // 经调试，这里的角度差存在判断错误的情况
    double delta_angle = angle_diff(curr_yaw, last_yaw);

    double w = delta_angle / dt;

    //std::cout << "v, w, dt:" << v << " " << w << " " << dt << std::endl;
    // 23.7.24 change to rosdebug
    RCLCPP_DEBUG(get_logger(), "v, w, dt:%f %f %f",v, w, dt);

    double delta_t = curRadarTime - last_local_time;
    PointTypePose keyPose;
    keyPose.yaw = last_yaw + delta_t * w;
    if (keyPose.yaw > M_PI)
      keyPose.yaw -= 2 * M_PI;
    else if (keyPose.yaw < -M_PI)
      keyPose.yaw += 2 * M_PI;

    keyPose.roll = curr_roll;
    keyPose.pitch = curr_pitch;
    //std::cout << "delta_t:" << delta_t << " interpolation yaw:" << keyPose.yaw << std::endl;
    // 23.7.24 change to rosdebug
    RCLCPP_DEBUG(get_logger(), "delta_t:%f interpolation yaw:%f",delta_t,  keyPose.yaw);

    double dx = v * cos(keyPose.yaw) * delta_t;
    double dy = v * sin(keyPose.yaw) * delta_t;

    keyPose.x = last_local_pose_x + dx;
    keyPose.y = last_local_pose_y + dy;
    keyPose.z = local_pose_.poses[bef_id].pose.position.z;

    // 重新赋值关键帧位姿
    (*cloudKeyPoses6D).points[ind].x = keyPose.x;
    (*cloudKeyPoses6D).points[ind].y = keyPose.y;
    (*cloudKeyPoses6D).points[ind].z = keyPose.z;
    (*cloudKeyPoses6D).points[ind].roll = keyPose.roll;
    (*cloudKeyPoses6D).points[ind].pitch = keyPose.pitch;
    (*cloudKeyPoses6D).points[ind].yaw = keyPose.yaw;
    (*cloudKeyPoses3D).points[ind].x = keyPose.x;
    (*cloudKeyPoses3D).points[ind].y = keyPose.y;
    (*cloudKeyPoses3D).points[ind].z = keyPose.z;
    return true;

  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
      return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }

  void publishFrames()
  {
      if (cloudKeyPoses3D->points.empty())
          return;
      // publish key poses
      publishCloud(pubKeyPoses, cloudKeyPoses3D, timeRadarInfoStamp, odometryFrame);
      
      if (pubRecentKeyFrame->get_subscription_count() != 0)
      {
          pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
          PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
          *cloudOut += *transformPointCloud(radarCloudRaw, &thisPose6D);

          publishCloud(pubRecentKeyFrame, cloudOut, timeRadarInfoStamp, odometryFrame);
      }
  } 

  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
  {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

      int cloudSize = cloudIn->size();
      cloudOut->resize(cloudSize);

      Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores) // 多线程并行
      for (int i = 0; i < cloudSize; ++i)
      {
          const auto &pointFrom = cloudIn->points[i];
          cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
          cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
          cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
          cloudOut->points[i].intensity = pointFrom.intensity;
      }
      return cloudOut;
  }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  void publishGlobalMap()
  {

      pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
      //pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

      // kd-tree to find near key frames to visualize
      std::vector<int> pointSearchIndGlobalMap;
      std::vector<float> pointSearchSqDisGlobalMap;

      for (int i = 0; i < (int)cloudKeyPoses3D->size(); ++i)
      {
          int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
          *globalMapKeyFrames += *transformPointCloud(allCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
      }
      sensor_msgs::msg::PointCloud2 cloudMsgGlobalMapKeyFrames;
      pcl::toROSMsg(*globalMapKeyFrames, cloudMsgGlobalMapKeyFrames);
      rclcpp::Clock clock;
      cloudMsgGlobalMapKeyFrames.header.stamp = clock.now();
      cloudMsgGlobalMapKeyFrames.header.frame_id = "odom";
      pubRadarCloudSurround->publish(cloudMsgGlobalMapKeyFrames);

  }

  void finishMappingSub(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_WARN(get_logger(), " mapping request finished by user!");
    // 保存地图
    rclcpp::Clock clock;
    rclcpp::Time begin_save_time = clock.now();
    bool is_save_map_succeed = false;
    while (!is_save_map_succeed && stamp2Sec(clock.now(), begin_save_time) < 5.0)
    {
      is_save_map_succeed = saveMap(map_save_dir, 0.0);
      rclcpp::sleep_for(std::chrono::milliseconds(200));
    }
    if (!is_save_map_succeed)
    {
      RCLCPP_ERROR(get_logger(), "in 5s, save point cloud map failure, please check your map_dir config or call save map by manual!");
    }
  }

  bool saveMap(std::string save_dir, float resolution)
  {
    string saveMapDirectory;
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    saveMapDirectory = std::getenv("HOME") + save_dir;
    cout << "Save destination: " << saveMapDirectory << endl;
    // create directory and remove old files;
    pcl::PointCloud<PointType>::Ptr globalAllCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalAllCloudDS(new pcl::PointCloud<PointType>());

    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
      // cout << "*********************************" << i << endl;
      *globalAllCloud += *transformPointCloud(allCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      // cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    if (resolution != 0)
    {
      cout << "\n\nSave resolution: " << resolution << endl;
      // down-sample and save corner cloud
      downSizeFilterCorner.setInputCloud(globalAllCloud);
      downSizeFilterCorner.setLeafSize(resolution, resolution, resolution);
      downSizeFilterCorner.filter(*globalAllCloudDS);
    }
    int ret = 0;
    // save global point cloud map
    *globalMapCloud += *globalAllCloud;

    // 半径滤波
    pcl::RadiusOutlierRemoval<PointType> outstrem;
    outstrem.setInputCloud(globalMapCloud);
    outstrem.setRadiusSearch(0.5); // 1m范围内至少有2个点
    outstrem.setMinNeighborsInRadius(5);
    // outstrem.setKeepOrganized(true);
    outstrem.filter(*globalMapCloud);
    // 统计滤波
    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(globalMapCloud);
    sor.setMeanK(5);
    sor.setStddevMulThresh(1.0);
    sor.filter(*globalMapCloud);

    ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
    return true;
  }

  void drivingMappingThread()
  {
    // 5s执行一次
    rclcpp::Rate rate(1);
    while (rclcpp::ok())
    {
      rate.sleep();
      // 时间戳对齐
      cloudQueueAlign();
      if(aloop_is_closed_ || allCloudKeyFrames.size()!=0)
      {
        updatePose();
      }
      publishGlobalMap();
    }
  }

  void visualizeGlobalMapThread()
  {
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }
  }


};

int main(int argc, char **argv)
{

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::executors::MultiThreadedExecutor exec;

  auto MO = std::make_shared<PointsFusion>(options);
  exec.add_node(MO);
  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> PointsFusion Started.\033[0m");
  std::thread drivingMapThread(&PointsFusion::drivingMappingThread, MO);
  std::thread visualizeMapThread(&PointsFusion::visualizeGlobalMapThread, MO);
  exec.spin();
  rclcpp::shutdown();

  drivingMapThread.join();

  return 0;
}
