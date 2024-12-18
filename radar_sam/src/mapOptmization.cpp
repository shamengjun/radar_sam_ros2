#include "utility.h"
#include "planar_factor.h"
#include "radar_sam/msg/cloud_info.hpp"
#include "radar_sam/srv/save_map.hpp"
#include "gpal_msgs/msg/vcu_data.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <pclomp/ndt_omp.h>
#include <GeographicLib/LocalCartesian.hpp> //包含头文件

using namespace gtsam;

// 应该是symbol的简写，用来索引变量
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

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

// 反正最终就是定义好了6自由度的位姿和其对应的时间
typedef PointXYZIRPYT PointTypePose;

class mapOptimization : public ParamServer
{

public:
  /**gtsam optimization**/
  NonlinearFactorGraph::shared_ptr gtSAMgraph; // gtsam factor图
  Values::shared_ptr initialEstimate;// 初始值，其实就是优化初值，之后需要用 .insert方法给这个变量赋值
  ISAM2 *isam;  // 调用isam
  Values::shared_ptr isamCurrentEstimate; // isam的当前值
  Key firstKey;  // first Key
  Eigen::MatrixXd poseCovariance; // 位姿协方差

  /**publisher**/
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubNewPath;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentSubmaps;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pubLoopClosed;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr notisfy_local_pub;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pubMapingStatus;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubOldPath;

  rclcpp::Service<radar_sam::srv::SaveMap>::SharedPtr srvSaveMap;
  rclcpp::Service<radar_sam::srv::SaveMap>::SharedPtr saveSemanticMapClient;

  /**subscriber**/
  rclcpp::Subscription<radar_sam::msg::CloudInfo>::SharedPtr subCloud;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPSRaw;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdometry;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr close_node_sub;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr finish_mapping_sub;
  rclcpp::Subscription<gpal_msgs::msg::VcuData>::SharedPtr subVcuData;
  // 存放vcuData话题中的数据
  std::deque<gpal_msgs::msg::VcuData> vcuDataQueue;
  // 存放档位id的队列
  std::vector<int> vcu_gear;
  std::deque<nav_msgs::msg::Odometry> gpsQueue;
  std::deque<sensor_msgs::msg::NavSatFix> gpsRawQueue; // add by zhaoyz 23.02.16
  std::deque<geometry_msgs::msg::PointStamped> gpsPointQueue;  // add by zhaoyz 23.6.2
  std::map<int, std::vector<double>> gps_key_map;  // double gps_lat, gps_long, gps_alt;
  std::deque<nav_msgs::msg::Odometry> imuQueue;
  std::deque<nav_msgs::msg::Odometry> wheelOdomQueue;

  radar_sam::msg::CloudInfo cloudInfo;

  // 存放关键帧的点云
  vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
  vector<pcl::PointCloud<PointType>::Ptr> allCloudKeyFrames;

  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointType>::Ptr newcloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
  pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D; 
  pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
  pcl::PointCloud<PointType>::Ptr mMcloudKeyPoses3D; // giseop
  pcl::PointCloud<PointTypePose>::Ptr mMcloudKeyPoses6D; // giseop
  map<int, pcl::PointCloud<PointTypePose>> mappingPath; // 存放每次的建图的关键帧位姿
  pcl::PointCloud<PointType>::Ptr laserCloudRaw; // giseop
  double laserCloudRawTime;

  pcl::PointCloud<PointType>::Ptr laserCloudAllLast; // surf feature set from odoOptimization
  pcl::PointCloud<PointType>::Ptr laserCloudAllLastDS; // downsampled surf featuer set from odoOptimization

  map<int, pcl::PointCloud<PointType>> laserCloudMapContainer;
  
  pcl::PointCloud<PointType>::Ptr laserCloudAllFromMap;
  pcl::PointCloud<PointType>::Ptr laserCloudAllFromMapDS;
  // add by yz.zhao for submap pointcloud

  pcl::PointCloud<PointType>::Ptr gpsKeyFramePose;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
  pcl::KdTreeFLANN<PointType>::Ptr kdtreeGpsKeyPoses;
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  pcl::VoxelGrid<PointType> downSizeFilterSurf;
  pcl::VoxelGrid<PointType> downSizeFilterICP;
  pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

  pcl::VoxelGrid<PointType> downSizeFilterSubmap;

  rclcpp::Time timeLaserInfoStamp;
  double timeLaserInfoCur;

  float transformTobeMapped[6];

  std::mutex mtx;
  std::mutex mtxLoopInfo;
  std::mutex odomMtx;
  std::mutex mtxGPS;
  std::mutex vcuMtx;
  std::mutex mtxGpsPose;

  bool isDegenerate = false;
  bool initial_position = true;
  int laserCloudAllLastDSNum = 0;
  bool positionDetectionFlag = false;
  bool aLoopIsClosed = false;
  bool is_new_trajectory = false; //24.1.12区分加载旧graph，新graph还没内和grapg产生关联的情况

  multimap<int, int> loopIndexContainer; // from new to old 

  vector<pair<int, int>> loopIndexQueue;
  vector<gtsam::Pose3> loopPoseQueue;
  vector<gtsam::SharedNoiseModel> loopNoiseQueue; // giseop for polymorhpisam (Diagonal <- Gausssian <- Base)
  deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

  nav_msgs::msg::Path globalOldPath;
  nav_msgs::msg::Path globalNewPath;
  nav_msgs::msg::Path oldRememberPath;

  Eigen::Affine3f transPointAssociateToMap;
  Eigen::Affine3f incrementalOdometryAffineFront;
  Eigen::Affine3f incrementalOdometryAffineBack;


  std::deque<pcl::PointCloud<PointType>> cloudQueue;
  int lastRSLoopId = 0;
  std::ofstream ofs;
  std::ofstream save_tum_pose;
  std::ofstream save_gps_tum_pose;
  std::ofstream save_trajectory;
  std::vector<double> keyFramePoseTimestamp;

  bool isLoopClosed = false;
  bool aLoopIsClosedOnlyICP = false; // 和gps添加到factor中
  int OldKeyFrameSize = 0;
  int valid_loop_num = 0;

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr loop_num_pub;

  GeographicLib::LocalCartesian geo_converter;
  bool rs_detect_loop = false;
  double curr_gps_z = 0.0;
  double last_gps_z = 0.0;
  double gps_convert_z = 0.0;
  std::vector<double> z_value_vec;
  std::vector<double> gps_z_value;

  std::ofstream graph_viz;

  int driveRounds = 0;
  int loopNums = 0;
  int i = 0;
  // 回环起点与终点
  pcl::PointCloud<PointType>::Ptr cloudLoopDis3D;
  // 记录gps起点终点
  pcl::PointCloud<PointType>::Ptr gpsLoop3D;
  double gpsMsgTime;
  bool firstLoop = true;
  double minDis = 0;
  // 记录时间戳
  double startTime = 0;
  double endTime = 0;

  // 记忆建图联合优化标记位
  bool MEMORY_MAAPPNG = false;
  bool LOAD_GPSPOSE = false;
  int id_gps_key_frame;
  // 发布建图状态
  enum MappingStatus
  {
    StartMapping,
    Mapping,
    EndMapping
  };

  std::unique_ptr<tf2_ros::TransformBroadcaster> br;
  mapOptimization(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_mapOptimization", options)
  {
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;  // isam2的参数设置 决定什么时候重新初始化
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    br = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/trajectory", 1);
    pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/map_global", 1);
    pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("radar_sam/mapping/odometry", 1);
    pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>(
        "radar_sam/mapping/odometry_incremental", qos);
    pubNewPath = create_publisher<nav_msgs::msg::Path>("radar_sam/mapping/path", 1);

    pubOldPath = create_publisher<nav_msgs::msg::Path>("radar_sam/mapping/remember_path", 1);

    // 非常重要的修改，我们直接去接特征提取前的cloud_info
    subCloud = create_subscription<radar_sam::msg::CloudInfo>(
        "radar_sam/deskew/cloud_info", qos,
        std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));

    subGPS = create_subscription<nav_msgs::msg::Odometry>(
        gpsTopic, 200,
        std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));

    subLoop = create_subscription<std_msgs::msg::Float64MultiArray>(
        "lio_loop/loop_closure_detection", qos,
        std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));

    subGPSRaw = create_subscription<sensor_msgs::msg::NavSatFix>(
        gps_topic_name, 10,
        std::bind(&mapOptimization::gpsRawInfoHandler, this, std::placeholders::_1));

    close_node_sub = create_subscription<std_msgs::msg::Empty>(
        "close_nodes", 1,
        std::bind(&mapOptimization::closeNodesSub, this, std::placeholders::_1));

    finish_mapping_sub = create_subscription<std_msgs::msg::Empty>(
        "finish_map", 1,
        std::bind(&mapOptimization::finishMappingSub, this, std::placeholders::_1));

    if (add_wheel_odometry)
    {
      subWheelOdometry = create_subscription<nav_msgs::msg::Odometry>(
          "rs/odom_a", 100,
          std::bind(&mapOptimization::wheelOdometryHandler, this, std::placeholders::_1));
    }

    subVcuData = create_subscription<gpal_msgs::msg::VcuData>(
        "vcu_data",100,
        std::bind(&mapOptimization::vcuDataHandler, this, std::placeholders::_1));

    pubLoopClosed = create_publisher<std_msgs::msg::Bool>("radar_sam/aloopIsClosed", 1);

    auto saveMapService = [this](const std::shared_ptr<rmw_request_id_t> request_header, const std::shared_ptr<radar_sam::srv::SaveMap::Request> req, std::shared_ptr<radar_sam::srv::SaveMap::Response> res) -> void
    {
      (void)request_header;
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if (req->destination.empty())
        saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else
        saveMapDirectory = std::getenv("HOME") + req->destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      // pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      for (int i = 0; i < (*cloudKeyPoses6D).size(); ++i)
      {
          save_trajectory.open(saveMapDirectory+"/trajectory.traj", std::ofstream::out | std::ofstream::app);
          save_trajectory.precision(18);
          double roll = (*cloudKeyPoses6D).points[i].roll;
          double pitch = (*cloudKeyPoses6D).points[i].pitch;
          double yaw = (*cloudKeyPoses6D).points[i].yaw;
          tf2::Quaternion quattf;
          quattf.setRPY(roll, pitch, yaw);

          geometry_msgs::msg::Quaternion quat;
          quat.x = quattf.x();
          quat.y = quattf.y();
          quat.z = quattf.z();
          quat.w = quattf.w();
          save_trajectory << keyFramePoseTimestamp[i] << "," << (*cloudKeyPoses6D).points[i].x  << "," << (*cloudKeyPoses6D).points[i].y << "," << 0 << ","
                                              <<  quat.x  << "," << quat.y << "," << quat.z << "," << quat.w << "," << vcu_gear[i]
                                              << std::endl;
          save_trajectory.close();
      }
      
      // add by zhaoyz 2022.09.22 保存tum格式的pose
      assert(cloudKeyPoses6D->size() == keyFramePoseTimestamp.size());

      if (is_save_tum_pose)
      {
        for (int i = 0; i < (*cloudKeyPoses6D).size(); ++i)
        {
          save_tum_pose << std::fixed << std::setprecision(6) << keyFramePoseTimestamp[i] << " "
                        << (*cloudKeyPoses6D).points[i].x << " " << (*cloudKeyPoses6D).points[i].y << " " << /*(*cloudKeyPoses6D).points[i].z*/ 0.0 << " ";
          double roll = (*cloudKeyPoses6D).points[i].roll;
          double pitch = (*cloudKeyPoses6D).points[i].pitch;
          double yaw = (*cloudKeyPoses6D).points[i].yaw;
          tf2::Quaternion quattf;
          quattf.setRPY(roll, pitch, yaw);

          geometry_msgs::msg::Quaternion quat;
          quat.x = quattf.x();
          quat.y = quattf.y();
          quat.z = quattf.z();
          quat.w = quattf.w();
          // save_tum_pose << quat.x << " " << quat.y << " " << quat.z << " " << quat.w << std::endl;
          save_tum_pose << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
        }
      }

      // pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalAllCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalAllCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
      {
        *globalAllCloud += *transformPointCloud(allCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if (req->resolution != 0)
      {
        cout << "\n\nSave resolution: " << req->resolution << endl;
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalAllCloud);
        downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
        downSizeFilterCorner.filter(*globalAllCloudDS);
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/AllMap.pcd", *globalAllCloudDS);
      }

      // save global point cloud map
      *globalMapCloud += *globalAllCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);

      res->success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n"
           << endl;

      // 保存初始时刻的gps
      std::ofstream save_gps_pose_ofs;
      // save_gps_pose_ofs.open(saveMapDirectory + "/initial_gps_pose.gp", std::ios::out | std::ios::app);
      // if (!save_gps_pose_ofs.is_open())
      //     RCLCPP_ERROR(get_logger(), "%s can't open!", (saveMapDirectory + "/initial_gps_pose.gp").c_str());
      //
      //                   << gps_lat << " " << gps_long  << " " << gps_alt;
      // save_gps_pose_ofs.close();
      save_gps_pose_ofs.open(saveMapDirectory + "/gps_keyframes_poses.gp", std::ios::out | std::ios::app);
      if (!save_gps_pose_ofs.is_open())
        RCLCPP_ERROR(get_logger(), "%s can't open!", (saveMapDirectory + "/gps_keyframes_poses.gp").c_str());
      save_gps_pose_ofs << "#gps pose align to map, keyframe_ind latitude longitude altitude." << std::endl;
      for (auto gps : gps_key_map)
        save_gps_pose_ofs << gps.first << " " << std::setprecision(12) << gps.second.at(0) << " " << gps.second.at(1) << " " << gps.second.at(2) << std::endl;
      save_gps_pose_ofs.close();
      if (mapping_mode == 2) // remember mode
      {
        std::cout << "mapping mode is remember mode." << std::endl;
        saveGraph(saveMapDirectory);
        saveKeyframePoints(saveMapDirectory);
        // printSaveInfo();
      }
      return;
    };

    srvSaveMap = create_service<radar_sam::srv::SaveMap>("radar_sam/save_map", saveMapService);

    pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/radar_sam/mapping/loop_closure_constraints", 1);

    pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/map_local", 1);
    pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/cloud_registered", 1);
    pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/cloud_registered_raw", 1);

    pubRecentSubmaps = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/mapping/submap", 1);

    notisfy_local_pub = create_publisher<std_msgs::msg::Empty>("program_run", 1);

    // 建图状态
    pubMapingStatus = create_publisher<std_msgs::msg::String>("radar_sam/mapping_status", 1);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    // 这里应该区分闭环和匹配时的降采样的大小
    downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingLoopLeafSize, mappingLoopLeafSize, mappingLoopLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    allocateMemory();

    // giseop
    if (is_save_align_info)
    {
      std::string save_align_info_path = getCurrentTime() + "_" + park_name + "_radar_sam_align_info.csv";
      ofs.open(save_align_info_path, std::ios::out | std::ios::app);
      if (!ofs.is_open())
        RCLCPP_ERROR(get_logger(), "Radar_slam_elapsed.csv open failure!");
      else
      {
        ofs << "ndt_match_time"
            << ","
            << "ndt_iter"
            << ","
            << "ndt_score" << std::endl;
      }
    }

    // 判断存在，先删除
    // int unused = system((std::string("exec rm -r ") /* + std::getenv("HOME") + "/.ros/" */ + tumPoseFile).c_str());
    if (is_save_tum_pose)
    {
      std::string tum_pose_file_path = getCurrentTime() + "_" + park_name + "_radar_sam_pose.txt";
      save_tum_pose.open(tum_pose_file_path, std::ios::out | std::ios::app);
      if (!save_tum_pose.is_open())
        RCLCPP_ERROR(get_logger(), "%s can't open!", tum_pose_file_path.c_str());
    }

    if (is_save_gps_pose)
    {
      std::string tum_pose_file_path = getCurrentTime() + "_" + park_name + "_gps_pose.txt";
      save_gps_tum_pose.open(tum_pose_file_path, std::ios::out | std::ios::app);
      if (!save_gps_tum_pose.is_open())
        RCLCPP_ERROR(get_logger(), "%s can't open!", tum_pose_file_path.c_str());
    }

    // graph_viz.open("radar_sam_graph.dot");
    gtSAMgraph.reset(new gtsam::NonlinearFactorGraph());
    initialEstimate.reset(new gtsam::Values());
    isamCurrentEstimate.reset(new gtsam::Values());
    if (mapping_mode == 2)
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> remember mapping mode.\033[0m");
      // 先检查路径是否存在，然后再调用load
      string saveMapDirectory;
      saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      std::string factor_graph_path = saveMapDirectory + "graph/slam_graph.txt";
      std::string trajectory_path = saveMapDirectory + "transformations.pcd";
      std::string clouds_path = saveMapDirectory + "all_clouds";
      ifstream ifs1(factor_graph_path.c_str());
      ifstream ifs2(trajectory_path.c_str());

      if (!ifs1 || !ifs2)
      {
        // throw invalid_argument("can not find file "  + factor_graph_path + " or " + trajectory_path);
        std::cout << "can not find file " + factor_graph_path + " or " + trajectory_path << std::endl;

        // isamCurrentEstimate.reset(new gtsam::Values());
        firstKey = 0;
      }
      else
      {
        if (!loadGraph(factor_graph_path, trajectory_path, clouds_path))
        {
          RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "load factor size is 0, this maybe first start remember!");
          firstKey = 0;
        }
      }
    }
    else
      firstKey = 0;
  }

  void allocateMemory()
  {
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    newcloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    mMcloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    mMcloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    cloudLoopDis3D.reset(new pcl::PointCloud<PointType>());
    gpsLoop3D.reset(new pcl::PointCloud<PointType>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeGpsKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    laserCloudRaw.reset(new pcl::PointCloud<PointType>()); // giseop

    laserCloudAllLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization

    laserCloudAllLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

    laserCloudAllFromMap.reset(new pcl::PointCloud<PointType>());
    
    gpsKeyFramePose.reset(new pcl::PointCloud<PointType>());
    laserCloudAllFromMapDS.reset(new pcl::PointCloud<PointType>());

    for (int i = 0; i < 6; ++i)
    {
      transformTobeMapped[i] = 0;
    }
  }

  bool loadGraph(const std::string &graph_file, const std::string &pose_file, std::string &pointcloud_path)
  {
    // 1. 从固定目录路径下找对应的graph
    std::cout << "graph_file:" << graph_file << std::endl;
    boost::tie(gtSAMgraph, initialEstimate) = readG2o(graph_file, true);
    std::cout << "load graph factor size:" << gtSAMgraph->size() << std::endl;
    std::cout << "load graph initial estimate value size:" << initialEstimate->size() << std::endl;
    std::cout << "last estimate value:" << initialEstimate->at<Pose3>(initialEstimate->size() - 1).translation().x() << " "
              << initialEstimate->at<Pose3>(initialEstimate->size() - 1).translation().y() << " "
              << initialEstimate->at<Pose3>(initialEstimate->size() - 1).translation().z() << std::endl;
    if (gtSAMgraph->size() == 0)
      return false;
    boost::shared_ptr<BetweenFactor<Pose3>> pose3Between =
        boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(gtSAMgraph->back());
    std::cout << "last key:" << pose3Between->key2() << std::endl;

    // 2. 用读到的transform.pcd填充, cloudKeyPoses6D,带时间戳,从transform.pcd中加载
    pcl::io::loadPCDFile<PointTypePose>(pose_file, *cloudKeyPoses6D);
    OldKeyFrameSize = cloudKeyPoses6D->size();

    // 先将文件夹中存下来的旧关键帧位姿存入mappingPath[0]中，后续对其进行更新
    mappingPath[0] = *cloudKeyPoses6D;
    // 3. 用cloudKeyPoses6D填充cloudKeyPoses3D
    for (int i = 0; i < cloudKeyPoses6D->size(); ++i)
    {
      PointType thisPoint;
      thisPoint.x = cloudKeyPoses6D->points[i].x;
      thisPoint.y = cloudKeyPoses6D->points[i].y;
      thisPoint.z = cloudKeyPoses6D->points[i].z;
      thisPoint.intensity = cloudKeyPoses6D->points[i].intensity;
      cloudKeyPoses3D->points.push_back(thisPoint);
      // if (i % 50 == 0)
      // {
      //   std::cout << cloudKeyPoses3D->points[i].x << " " << cloudKeyPoses3D->points[i].y << " " << cloudKeyPoses3D->points[i].z
      //             << " " << cloudKeyPoses3D->points[i].intensity << std::endl;
      // }
      // 3.2 填充oldRememberPath
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.stamp = rclcpp::Time(cloudKeyPoses6D->points[i].time);
      pose_stamped.header.frame_id = odometryFrame;
      pose_stamped.pose.position.x = thisPoint.x;
      pose_stamped.pose.position.y = thisPoint.y;
      pose_stamped.pose.position.z = thisPoint.z;
      tf2::Quaternion q;
      q.setRPY(cloudKeyPoses6D->points[i].roll, cloudKeyPoses6D->points[i].pitch, cloudKeyPoses6D->points[i].yaw);
      pose_stamped.pose.orientation.x = q.x();
      pose_stamped.pose.orientation.y = q.y();
      pose_stamped.pose.orientation.z = q.z();
      pose_stamped.pose.orientation.w = q.w();

      oldRememberPath.poses.push_back(pose_stamped);
    }
    // 4. 将轨迹以及点云可视化出来
    // 后续应该可以发出来。
    // 5. 填充factor的key.即改变key
    firstKey = cloudKeyPoses3D->points.size();

    // 6. 填充allCloudKeyFrames
    for (int i = 0; i < cloudKeyPoses6D->size(); ++i)
    {
      std::string curr_file_path = pointcloud_path + "/" + std::to_string(i) + ".pcd";

      pcl::PointCloud<PointType>::Ptr curr_points(new pcl::PointCloud<PointType>());
      pcl::io::loadPCDFile<PointType>(curr_file_path, *curr_points);
      allCloudKeyFrames.push_back(curr_points);
    }

    // 7. 调用一下优化
    if (is_global_optimization)
    {
      gtsam::LevenbergMarquardtParams lmparameters;
      lmparameters.relativeErrorTol = 1e-5;
      lmparameters.maxIterations = 100; // 这里的最大迭代次数可能需要时间的消耗调整
      gtsam::LevenbergMarquardtOptimizer optimizer(*gtSAMgraph, *initialEstimate, lmparameters);
      *isamCurrentEstimate = optimizer.optimize();
      // gtsam::Marginals marginals(*gtSAMgraph, *isamCurrentEstimate);
      // std::cout << "optimizaion: 4" << std::endl;
      // poseCovariance = marginals.marginalCovariance(isamCurrentEstimate->size() - 1);
    }

    return true;
  }

  // 读去gp文件中与关键帧对应的gps数值
  bool loadGpsPose(const std::string &gps_path)
  {
    mtxGpsPose.lock();
    std::cout << "gps_file:" << gps_path << std::endl;
    // 将gp文件中的数据存入PointType类型数据
    std::ifstream gps_file(gps_path + "/gps_keyframes_poses.gp");
    std::string gps_line;
    if (!gps_file.is_open()) 
    {
      std::cerr << "Error: Unable to open GPS file" << std::endl;
      return false;
    }
    while(std::getline(gps_file, gps_line))
    {
      if(gps_line.empty())
      {
        continue;
      }
      std::cout << "gps_line:" << gps_line<< std::endl;
      std::istringstream iss(gps_line);
      PointType point;
      iss >> point.intensity >> point.y >> point.x >> point.z;
      point.z = 0;
      std::cout << "point.intensity:" << point.intensity << std::endl;
      gpsKeyFramePose->push_back(point);
      std::cout << "gpsKeyFramePose->size():" << gpsKeyFramePose->size() << std::endl;
    }
    return true;
    mtxGpsPose.unlock();
  }
  void publishMappingStatus(MappingStatus mappingStatus, int pub_num)
  {
    std_msgs::msg::String mappingStatusMessages;
    string mappingStatu;
    // ros::Rate pub_mappingstatus(10);
    switch (mappingStatus)
    {
    case StartMapping:
      mappingStatu = "[SLAM]_started";
      mappingStatusMessages.data = mappingStatu;
      break;
    case Mapping:
      mappingStatu = "[SLAM]_mapping!";
      mappingStatusMessages.data = mappingStatu;
      break;
    case EndMapping:
      mappingStatu = "[SLAM]_finished";
      mappingStatusMessages.data = mappingStatu;
      break;
    default:
      break;
    }

    for (int i = 0; i < pub_num; ++i)
    {
      pubMapingStatus->publish(mappingStatusMessages);
      if (pub_num > 1)
        rclcpp::sleep_for(std::chrono::microseconds(100));
    }
  }

  // 这里写的subscriber的queue的size为1，也就说处理当前最新的数据，当这个回调函数
  // 处理跟不上时，可能会丢帧
  void laserCloudInfoHandler(const radar_sam::msg::CloudInfo::SharedPtr msgIn)
  {
    // 这里提取了msg的header的时间戳和 toSec值，应该是一样的只是单位不同
    timeLaserInfoStamp = msgIn->header.stamp;
    timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

    // extract info and feature cloud
    cloudInfo = *msgIn;

    pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudAllLast);
    pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudRaw); // giseop
  

    // startmapping and mapping status
    if (!laserCloudRaw->points.empty())
    {
      static bool is_mapping_started = false;
      if (allCloudKeyFrames.size() < 2 && !is_mapping_started)
      {
        MappingStatus currentStatus = StartMapping;
        publishMappingStatus(currentStatus, 5); // 10
        is_mapping_started = true;
      }
      else
      {
        MappingStatus currentStatus = Mapping;
        publishMappingStatus(currentStatus, 1);
      }
    }

    std::lock_guard<std::mutex> lock(mtx);

    static double timeLastProcessing = -1;
    // 后面这个值默认是0.15 第一帧肯定通过，之后把timeLastProcessing设置成上一帧的时间，看当前帧时间和上一帧时间差是否大于0.15
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
    {
      timeLastProcessing = timeLaserInfoCur;
      // 23.7.12 如果点的个数过少，直接不进行以下处理
      //1. 首先需要标识当前的trajectory_id为0还是1
#if 0
    if (laserCloudRaw->points.size() < 400 && cloudKeyPoses3D->points.size() > 2)
    {
      RCLCPP_WARN(get_logger(), "pointcloud size is less than 300, does not match !");
      publishOdometry();
      return;
    }
#endif
      if(!MEMORY_MAAPPNG && (firstKey != 0))
      {
        std::cout << "Not near the memory lane!" << std::endl;
        return;
      }
      // std::cout << "gpsRawQueue size: " << gpsRawQueue.size() << std::endl;
      // 使用imu和odom的信息来设置初始估计，结果都放在transformTobeMapped里
      updateInitialGuess();

      if(initial_position)
      {
        return;
      }

      extractSurroundingKeyFrames();

      downsampleCurrentScan();

      scan2MapOptimization();

      saveKeyFramesAndFactor();

      correctPoses();

      publishOdometry();

      publishFrames();
    }
    else
    {
    }
  }

  // add by zhaoyz 22.11.29
  void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr imuMsg)
  {
    // std::lock_guard<std::mutex> lock2(odoLock);
    imuQueue.push_back(*imuMsg);
    // 这里只保留当前时刻1s内的imu数据
  }

  // add by zhaoyz 22.11.30 这个wheel odometry是高频的，和imu频率一致为100hz
  // 在save keyframe后记录下当前对应的wheel odom值
  void wheelOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
  {
    wheelOdomQueue.push_back(*odomMsg);
  }

  void vcuDataHandler(const gpal_msgs::msg::VcuData::SharedPtr vcuDataMsg)
  {
      std::lock_guard<std::mutex> lock(vcuMtx);
      vcuDataQueue.push_back(*vcuDataMsg);
  }

  int associateKeyframeWithVcuDAata(double time)
  {
      std::lock_guard<std::mutex> lock(vcuMtx);
      if(vcuDataQueue.empty())
      {
          RCLCPP_WARN(this->get_logger(), "vcuDataQueue is empty!");
          return 0;
      }
      // 进行时间同步 这里的VcuData频率较高
      while(!vcuDataQueue.empty())
      {
          if (stamp2Sec(vcuDataQueue.front().header.stamp) < time)
          {
              vcuDataQueue.pop_front();
          }
          else
          {
              return vcuDataQueue.front().drive_mode;  
              // std::cout << "associateKeyframeWithVcuDAata id: " << vcuDataQueue.front().drive_mode << std::endl;
              break;              
          }
      }
  }

  void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
  {
    gpsQueue.push_back(*gpsMsg);
    nav_msgs::msg::Odometry msgGPS;
    msgGPS = *gpsMsg;
    gpsMsgTime = stamp2Sec(msgGPS.header.stamp);
    float gpsx = msgGPS.pose.pose.position.x;
    float gpsy = msgGPS.pose.pose.position.y;
    float gpsz = 0;
    PointType curGPSInf;
    curGPSInf.x = gpsx;
    curGPSInf.y = gpsy;
    curGPSInf.z = gpsz;
    curGPSInf.intensity = gpsLoop3D->size();
    gpsLoop3D->push_back(curGPSInf);
  }

  void gpsRawInfoHandler(const sensor_msgs::msg::NavSatFix::SharedPtr gps_msg)
  {
    mtxGPS.lock();
    // gpsRawQueue.push_back(*gps_msgs);

    // if (gps_msg->status.status != 69 && gps_msg->status.status != 75 && gps_msg->status.status != 85 && gps_msg->status.status != 91)
    // {
    //     RCLCPP_ERROR(get_logger(), "gps staus %d is not availble.", gps_msg->status.status);
    //     // return;
    // }
    // else
    // {
    curr_gps_z = gps_msg->altitude;
    gpsRawQueue.push_back(*gps_msg);
    static bool first_keyframe = true;

    if (first_keyframe)
    {
      geo_converter.Reset(gps_msg->latitude, gps_msg->longitude, gps_msg->altitude);
      first_keyframe = false;
    }
    double x, y, z;
    geo_converter.Forward(gps_msg->latitude, gps_msg->longitude, gps_msg->altitude, x, y, z);
    if (is_save_gps_pose)
      save_gps_tum_pose << std::fixed << std::setprecision(6) << stamp2Sec(gps_msg->header.stamp) << " " << x << " " << y << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
    geometry_msgs::msg::PointStamped gps_p;
    gps_p.header.stamp = gps_msg->header.stamp;
    gps_p.point.x = x;
    gps_p.point.y = y;
    gps_p.point.z = z;
    gpsPointQueue.push_back(gps_p);
    // }
    // 2023.5.10 save gps pose
    //  队列维护近3s内的数据
    // while ((stamp2Sec(gpsRawQueue.back().header.stamp) - stamp2Sec(gpsRawQueue.front().header.stamp)) > 3.0)
    //   gpsRawQueue.pop_front();
    mtxGPS.unlock();
  }

  bool getCurrGPSPoint(double t, geometry_msgs::msg::PointStamped &gps)
  {
    if (gpsPointQueue.empty())
      return false;
    while (!gpsPointQueue.empty())
    {
      if (stamp2Sec(gpsPointQueue.front().header.stamp) < t)
        gpsPointQueue.pop_front();
      else
      {
        gps = gpsPointQueue.front();
        return true;
      }
    }
    return false;
  }

  void pointAssociateToMap(PointType const *const pi, PointType *const po)
  {
    po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
    po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
    po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
    po->intensity = pi->intensity;
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

  pcl::PointCloud<PointType>::Ptr transformPointClouduseTransformTobeMapped(pcl::PointCloud<PointType>::Ptr cloudIn)
  {
    updatePointAssociateToMap(); // 这里已经把当前帧imu预积分添加进去了，即已经更新了transformTobeMapped这个变量了

    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; i++)
    {
      PointType pointOri, pointSel;
      pointOri = cloudIn->points[i];
      // 将每个点转到地图系，也即绝对坐标
      pointAssociateToMap(&pointOri, &pointSel);
      cloudOut->points[i] = pointSel;
    }
    return cloudOut;
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
  }

  gtsam::Pose3 trans2gtsamPose(float transformIn[])
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }

  Eigen::Affine3f trans2Affine3f(float transformIn[])
  {
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
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

  void visualizeGlobalMapThread()
  {
    // 5s执行一次
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok())
    {
      rate.sleep();
      publishGlobalMap();

      // publish path
      if (pubOldPath->get_subscription_count() != 0)
      {
        oldRememberPath.header.stamp = this->get_clock()->now();
        oldRememberPath.header.frame_id = odometryFrame;
        pubOldPath->publish(oldRememberPath);
      }
    }

    if (savePCD == false)
      return;

    // save map
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // save key frame transformations
    pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
    // extract global point cloud map
    pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());

    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
      *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
    }
    // down-sample and save corner cloud
    downSizeFilterCorner.setInputCloud(globalCornerCloud);
    downSizeFilterCorner.filter(*globalCornerCloudDS);
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
    // down-sample and save global point cloud map
    *globalMapCloud += *globalCornerCloud;
    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed" << endl;
  }

  void publishGlobalMap()
  {
    if (pubLaserCloudSurround->get_subscription_count() == 0)
      return;

    if (cloudKeyPoses3D->points.empty() == true)
      return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    // pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                            // for global map visualization
    // downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    // downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    // downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // for (auto &pt : globalMapKeyPosesDS->points)
    // {
    //   kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
    //   // pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    // }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPoses->size(); ++i)
    {
      // if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
      //     continue;
      int thisKeyInd = (int)globalMapKeyPoses->points[i].intensity;

      *globalMapKeyFrames += *transformPointCloud(allCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

    publishCloud(pubLaserCloudSurround, globalMapKeyFrames, timeLaserInfoStamp, odometryFrame);
  }

  void loopClosureThread()
  {
    if (loopClosureEnableFlag == false)
      return;

    rclcpp::Rate rate(loopClosureFrequency);
    while (rclcpp::ok())
    {
      rate.sleep();

      performRSLoopClosure();
      visualizeLoopClosure();
    }
  }

  void positionDetThread()
  {
    // 当检测到当前位置就在旧轨迹附近时，不进行下面的操作
    if (positionDetectionFlag == true)
    {
        return;
    }

    // 读取旧轨迹对应的gps值
    string saveMapDirectory;
    saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    if(!loadGpsPose(saveMapDirectory))
    {
        LOAD_GPSPOSE = false;
    }
    else
    {
        // 若load到旧gps轨迹，则进行下面的操作
        LOAD_GPSPOSE = true;

        // 一直去计算当前gps与旧轨迹gps的距离
        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();
            positionDetClosure();
            if(positionDetectionFlag)
            {
                break;
            }
        }  
    }

  }

  void positionDetClosure()
  {
    // 若是正常建图状态，也就是历史关键帧队列size为0，则不进行操作，否则就开始寻找距离最近的关键帧
    if(firstKey == 0)
    {
        positionDetectionFlag = true;
        return;
    }
    else
    {
        // 在当前gps队列中没有数时，先不进行操作
        if (gpsRawQueue.empty())
        {
            std::cout << "===>positionDetThread: wait for gps!" << std::endl;
            return;
        }
        std::cout << "===>positionDetClosure start!" << std::endl;
        // 根据当前gps的值寻找上一次记忆路线的关键帧位姿 == 粗定位
        sensor_msgs::msg::NavSatFix gps_msg = gpsRawQueue.back();
        PointType cur_point;
        cur_point.x = gps_msg.longitude;
        cur_point.y = gps_msg.latitude;
        cur_point.z = 0;
        // std::cout << "cur_point.x: " << cur_point.x << " cur_point.y: " << cur_point.y << " cur_point.z: " << cur_point.z << std::endl;
        // 创建KD树
        kdtreeGpsKeyPoses->setInputCloud(gpsKeyFramePose);
        std::vector<int> pointIdxRadiusSearch(1);
        std::vector<float> pointRadiusSquaredDistance(1);
        // 寻找距离当前gps点最近的点
        kdtreeGpsKeyPoses->radiusSearch(cur_point, 1.0, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        int id = pointIdxRadiusSearch[0];
        id_gps_key_frame = id;
        double distance = pointRadiusSquaredDistance[id];
        PointType a_point = gpsKeyFramePose->points[id];
        // std::cout << "===>positionDetClosure: a_point.x- cur_point.x" << 
        //     a_point.x - cur_point.x << " a_point.y- cur_point.y" << 
        //     a_point.y - cur_point.y << std::endl;
        // std::cout << "===>positionDetThread: distance: " << distance << std::endl;

        // 一但检测到当前位置与找到的历史关键帧的位置距离小与0.001，则认为找到了，可以开始多轨迹联合优化建图，并不再继续进行该线程中的操作
        if(distance < 5e-8)
        {
            positionDetectionFlag = true;
            MEMORY_MAAPPNG = true;
        }
    }
  }

  void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg)
  {
    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopMsg->data.size() != 2)
      return;

    loopInfoVec.push_back(*loopMsg);

    while (loopInfoVec.size() > 5)
      loopInfoVec.pop_front();
  }

  void performRSLoopClosure()
  {
    if (cloudKeyPoses3D->points.empty() == true)
      return;

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    copy_cloudKeyPoses2D->clear();            // giseop
    *copy_cloudKeyPoses2D = *cloudKeyPoses3D; // giseop
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false) // 订阅外部模块闭环检测的信息
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    // std::cout << "RS loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop
    RCLCPP_INFO(get_logger(), "RS loop found! between %d and %d", loopKeyCur, loopKeyPre);

    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
      loopFindCurNearKeyframes(cureKeyframeCloud, loopKeyCur, 1);
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
      if (cureKeyframeCloud->size() < 200 || prevKeyframeCloud->size() < 1000)
        return;
      if (pubHistoryKeyFrames->get_subscription_count() != 0)
        publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(200);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
    {
      // std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this RS loop." << std::endl;
      RCLCPP_DEBUG(get_logger(), "ICP fitness test failed (%f > %f). Reject this RS loop.", icp.getFitnessScore(), historyKeyframeFitnessScore);
      rs_detect_loop = false;
      return;
    }
    else
    {
      // std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this RS loop." << std::endl;
      RCLCPP_DEBUG(get_logger(), "ICP fitness test passed (%f < %f). Add this RS loop.", icp.getFitnessScore(), historyKeyframeFitnessScore);
      rs_detect_loop = true;
    }

    // publish corrected cloud
    if (pubIcpKeyFrames->get_subscription_count() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
      publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    if(mapping_mode == 2 && OldKeyFrameSize != 0)
    {
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-2;
    }
    else
    {
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;        
    }

    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    // loopIndexContainer[loopKeyCur] = loopKeyPre;
    loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
  } // performRSLoopClosure

  bool detectLoopClosureDistance(int *latestID, int *closestID)
  {
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end()) // 找到了，之前已经添加过了，所以不用再添加了
      return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;

    // copy_cloudKeyPoses2D仅仅把z轴进行了限制，然后在其附近进行半径搜索？这样可以吗？待测试 zhaoyz 07.05 comment
    //  for (int i = 0; i < (int)copy_cloudKeyPoses2D->size(); i++) // giseop
    //      copy_cloudKeyPoses2D->points[i].z = 2.0;                // to relieve the z-axis drift, 1.1 is just foo val

    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D);                                                                                  // giseop
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0); // giseop

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
      {
        loopKeyPre = id;
        break;
      }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
      return false;

    // 判断loopKeyPre是否离得太近 zhaoyz
    //  if (lastRSLoopId > 0)
    //  {
    //     if (loopKeyPre - lastRSLoopId < loopIntervalFrames)
    //      return false;
    //  }

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    // lastRSLoopId = loopKeyPre;
    if (abs(loopKeyCur - loopKeyPre) < loopIntervalFrames)
    {
      RCLCPP_WARN(get_logger(), "loop %d and %d is a bit closing, this maybe is not a valid loop!", loopKeyPre, loopKeyCur);
      return false;
    }

    return true;
  }

  bool detectLoopClosureExternal(int *latestID, int *closestID)
  {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtxLoopInfo);
    if (loopInfoVec.empty()) // 外部传进来的闭环检测信息
      return false;

    double loopTimeCur = loopInfoVec.front().data[0];
    double loopTimePre = loopInfoVec.front().data[1];
    loopInfoVec.pop_front();

    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
      return false;

    int cloudSize = copy_cloudKeyPoses6D->size();
    if (cloudSize < 2)
      return false;

    // latest key
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i)
    {
      if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
        loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i)
    {
      if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
        loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
      else
        break;
    }

    if (loopKeyCur == loopKeyPre)
      return false;

    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize)
        continue;
      *nearKeyframes += *transformPointCloud(allCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
      return;
    // if (searchNum == 0)
    //    return;
    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  void loopFindCurNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    if(key > OldKeyFrameSize + searchNum)
    {
        for(int i = searchNum; i >0; i--)
        {
            int keyNear = cloudSize - i;
            std::cout << "keyNear: " << keyNear << std::endl;
            *nearKeyframes += *transformPointCloud(allCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);

        }
    }
    else
    {
        *nearKeyframes = *transformPointCloud(allCloudKeyFrames[key], &copy_cloudKeyPoses6D->points[key]);
    }

    if (nearKeyframes->empty())
      return;
    // if (searchNum == 0)
    //    return;
    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }


  void visualizeLoopClosure()
  {
    if (loopIndexContainer.empty())
      return;

    visualization_msgs::msg::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::msg::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::msg::Marker::ADD;
    markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::msg::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::msg::Marker::ADD;
    markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::msg::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge->publish(markerArray);
  }

  void updateInitialGuess()
  {
    // save current transformation before any processing
    incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

    static Eigen::Affine3f lastImuTransformation;
    // initialization 初始化
    if (cloudKeyPoses3D->points.empty())
    {
      transformTobeMapped[0] = 0;
      transformTobeMapped[1] = 0;
      transformTobeMapped[2] = cloudInfo.imu_yaw_init;

      // 下面这个值默认是false 所以if总是将transformTobeMapper[2]设置位0
      if (!useImuHeadingInitialization)
        transformTobeMapped[2] = 0;
      transformTobeMapped[3] = 0;
      transformTobeMapped[4] = 0;
      transformTobeMapped[5] = 0;
      lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
      initial_position = false;
      return;
    }
    else if(initial_position && mapping_mode == 2)
    {

      RCLCPP_INFO(get_logger(), "\033[1;34m---->cloudKeyPoses3D->points.size() == firstKey \033[0m");
      // 初始值设为在之前路径中找到的位姿
      /*
        1. 判断当前gps是否有值
        2. 根据gps值在gp文件中寻找key_frame的id
        3. 根据id在关键帧位姿中寻找对应位姿
        4. 将对应的位姿赋值给transformTobeMapped作为初始值
      */
      if(gpsRawQueue.empty())
      {
        return;
      }
      else
      {
        if(!LOAD_GPSPOSE)
        {
          transformTobeMapped[3] = 0;
          transformTobeMapped[4] = 0;
          transformTobeMapped[5] = 0;
          transformTobeMapped[0] = cloudInfo.imu_roll_init;
          transformTobeMapped[1] = cloudInfo.imu_pitch_init;
          transformTobeMapped[2] = cloudInfo.imu_yaw_init;
          RCLCPP_INFO(get_logger(), "\033[1;34m---->cloudInfo! cloudInfo.imu_roll_init: %d, cloudInfo.imu_pitch_init: %d, cloudInfo.imu_yaw_init: %d, \033[0m", cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
          // 下面这个值默认是false 所以if总是将transformTobeMapper[2]设置位0
          if (!useImuHeadingInitialization)
            transformTobeMapped[2] = 0;

          lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
          initial_position = false;
          return;
        }
        else
        {
          int id = id_gps_key_frame;
          PointType a_point = gpsKeyFramePose->points[id];
          int cur_key_frame_id = a_point.intensity;
          std::cout << "cur_key_frame_id: " << cur_key_frame_id << std::endl;
        //   RCLCPP_INFO(rclcpp::get_logger("Memory Mapping Position"), "\033[1;32m----> distance : %.7f \033[0m",pointRadiusSquaredDistance[id]);
        //   std::cout << "distance: " << pointRadiusSquaredDistance[0] << std::endl;

          // 根据这个id在关键帧队列中找到对应的位姿并赋给初始位姿
          if (cur_key_frame_id != -1)
          {
            transformTobeMapped[3] = cloudKeyPoses6D->points[cur_key_frame_id].x;
            transformTobeMapped[4] = cloudKeyPoses6D->points[cur_key_frame_id].y;
            transformTobeMapped[5] = cloudKeyPoses6D->points[cur_key_frame_id].z;
            transformTobeMapped[0] = cloudKeyPoses6D->points[cur_key_frame_id].roll;
            transformTobeMapped[1] = cloudKeyPoses6D->points[cur_key_frame_id].pitch;
            transformTobeMapped[2] = cloudKeyPoses6D->points[cur_key_frame_id].yaw;
            lastImuTransformation = pcl::getTransformation(transformTobeMapped[3],transformTobeMapped[4],transformTobeMapped[5],transformTobeMapped[0],transformTobeMapped[1],transformTobeMapped[2]); // save imu before return;
          }
          initial_position = false;
        }
      }
      return;
    }

    // 使用imu预积分来做位姿猜测值
    // use imu pre-integration estimation for pose guess
    static bool lastImuPreTransAvailable = false;
    static Eigen::Affine3f lastImuPreTransformation;

    // 这个值是在imageProjection中设置，如果odom经过了那边的判断而可用，就设置true
    if (cloudInfo.odom_available == true)
    {
      // std::cout << "cloudInfo.odom_available == true" << std::endl;
      //  当前点云帧的pose initial guess
      Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                                                         cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
      if (lastImuPreTransAvailable == false)
      {
        lastImuPreTransformation = transBack;
        lastImuPreTransAvailable = true;
      }
      else
      {
        // 当前的odom给出的transBack和上一帧的trans的逆相乘
        Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
        // 这一帧的transTobeMap 来自于imu_roll_init 其实就是当前的imu信息
        Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
        // 相乘
        Eigen::Affine3f transFinal = transTobe * transIncre;
        // 结果放在transTobeMapped里
        pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        lastImuPreTransformation = transBack;

        lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;

        return;
      }
    }

    // 同样，来源于imageProjection中的判定imu可不可用
    //  use imu incremental estimation for pose guess (only rotation)
    // if (cloudInfo.imu_available == true ) // 以下部分应该只执行一次
    // {
    //   Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
    //   Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;
    //   //@todo需要转换到map坐标系
    //   increTrans = transIncre;
    //   Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
    //   Eigen::Affine3f transFinal = transTobe * transIncre;
    //   pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

    //   lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
    //   return;
    // }
  }

  void extractForLoopClosure()
  {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
    int numPoses = cloudKeyPoses3D->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
      if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
        cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
      else
        break;
    }

    extractCloud(cloudToExtract);
  }

  void extractNearby()
  {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // 记忆建图在非初次进行建图时，要与之前的关键帧进行区别，不能在过去的关键帧中寻找prePointCloud  TODO:
    if(firstKey == 0)
    {
      // extract all the nearby key poses and downsample them  找50m范围内的关键帧
      kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tre
      // surroundingKeyframeSearchRadius 默认设置位50
      // 搜索结果放在pointSearchInd, pointSearchSqDis  注意 后者是squared distance的意思 也即距离平方
      kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
      // 遍历搜索结果，将搜索结果的点云帧加入到surroundingKeyPoses里
      for (int i = 0; i < (int)pointSearchInd.size(); ++i)
      {
        int id = pointSearchInd[i];
        surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
      }
      // debug下这里的降采样的输出,这里的降采样还是有效的
      downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
      // surroundingKeyPosesDS中存放下采样结果，这里我们看得到，这个滤波器的参数是1.0，可能意味着这里其实没有做下采样?并不是的
      downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

      // 把10s内同方向的关键帧也加到surroundingKeyPosesDS中，防止车同向一直旋转
      // also extract some latest key frames in case the robot rotates in one position
      int numPoses = cloudKeyPoses3D->size();
      for (int i = numPoses - 1; i >= 0; --i)
      {
        if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
          surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
        else
          break;
      }
      // 有了周围的关键帧了，接下来提取关键帧对应的点云
      extractCloud(surroundingKeyPosesDS);      
    }
    else
    {
      // 说明是第二次进行记忆建图
      // extract all the nearby key poses and downsample them  找50m范围内的关键帧
      kdtreeSurroundingKeyPoses->setInputCloud(mMcloudKeyPoses3D); // create kd-tre
      // surroundingKeyframeSearchRadius 默认设置位50
      // 搜索结果放在pointSearchInd, pointSearchSqDis  注意 后者是squared distance的意思 也即距离平方
      kdtreeSurroundingKeyPoses->radiusSearch(mMcloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
      // 遍历搜索结果，将搜索结果的点云帧加入到surroundingKeyPoses里
      for (int i = 0; i < (int)pointSearchInd.size(); ++i)
      {
        int id = pointSearchInd[i];
        surroundingKeyPoses->push_back(mMcloudKeyPoses3D->points[id]);
      }
      // debug下这里的降采样的输出,这里的降采样还是有效的
      downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
      // surroundingKeyPosesDS中存放下采样结果，这里我们看得到，这个滤波器的参数是1.0，可能意味着这里其实没有做下采样?并不是的
      downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

      // 把10s内同方向的关键帧也加到surroundingKeyPosesDS中，防止车同向一直旋转
      // also extract some latest key frames in case the robot rotates in one position
      int numPoses = mMcloudKeyPoses3D->size();
      for (int i = numPoses - 1; i >= 0; --i)
      {
        if (timeLaserInfoCur - mMcloudKeyPoses6D->points[i].time < 10.0)
          surroundingKeyPosesDS->push_back(mMcloudKeyPoses3D->points[i]);
        else
          break;
      }
      // 有了周围的关键帧了，接下来提取关键帧对应的点云
      extractCloud(surroundingKeyPosesDS);     
    }

  }

  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) // 这里传进来的是surroundingKeyPoseDS即cloudKeyPoses3D->back()附近50m范围内的关键帧pose
  {
    // fuse the map 添加我们的点云
    laserCloudAllFromMap->clear();

    // 遍历所有帧pose
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
      // 如果遍历的帧和当前帧的距离太大，直接continue，不知道这步有什么意义 这里的surroudingKeyframesSearchRadius是50m
      // 这里似乎不会，因为在extractNearby()函数已经半径搜索过了
      // if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
      //   continue;

      // 这里intensity，实际上本来是index，是在（thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index）里赋值的
      // 所以thisKeyInd为，当前遍历帧的索引（在位姿序列中）
      int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      // 如果在laserCloudMapContainer中，一开始这个变量为空，肯定不在
      // 这个的目的是，如果当前遍历的帧之前遍历过了（有历史信息），就不要再计算变换了，太浪费算力
      // 直接修改这个Container，我们没有两种不同的点云，不需要用pair来存
      if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end())
      {
        *laserCloudAllFromMap += laserCloudMapContainer[thisKeyInd];
      }
      else
      {
        // 进行坐标变换
        pcl::PointCloud<PointType> laserCloudAllTemp = *transformPointCloud(allCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *laserCloudAllFromMap += laserCloudAllTemp;

        // 最后保存，进行修改
        laserCloudMapContainer[thisKeyInd] = laserCloudAllTemp;
      }
    }

    downSizeFilterSurf.setInputCloud(laserCloudAllFromMap);
    downSizeFilterSurf.filter(*laserCloudAllFromMapDS);

    // clear map cache if too large
    if (laserCloudMapContainer.size() > 1000)
      laserCloudMapContainer.clear();
  }

  void extractSurroundingKeyFrames()
  {
    // 第一帧直接return
    if (cloudKeyPoses3D->points.empty() == true || cloudKeyPoses3D->points.size() == firstKey)
    {
      std::cout << "[extractSurroundingKeyFrames] first frame ---> return;" << std::endl;
      return;
    }

    extractNearby();
  }

  void downsampleCurrentScan()
  {
    // 我们不一定需要下采样 这里下采样参数是0.2
    // 这种采样方法是求处在同一体素内的3d点的中心点作为该体素内的唯一一个3d点，在0.2的参数下，对于毫米波，应该开不开都无所谓
    laserCloudAllLastDS->clear();
    downSizeFilterSurf.setInputCloud(laserCloudAllLast);
    downSizeFilterSurf.filter(*laserCloudAllLastDS);
    laserCloudAllLastDSNum = laserCloudAllLastDS->size();
  }

  void updatePointAssociateToMap()
  {
    transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
  }

  void scan2MapOptimization()
  {
    if (cloudKeyPoses3D->points.empty() || mMcloudKeyPoses3D->points.empty())
    {
      std::cout << "[scan2MapOptimization] first frame--->return" << std::endl;
      return;
    }

    // std::cout << "laser cloud ds num:" << laserCloudAllLastDSNum << std::endl;
    if (laserCloudAllLastDSNum > 50) // 原来是 线特征多于10个 平面特征多于100个 我们的话 直接大于100吧 100-->60
    {
      // 获取当前帧点云
      pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr cureKeyframeCloudTmp(new pcl::PointCloud<PointType>());
      // pcl::copyPointCloud(*laserCloudAllLastDS, *cureKeyframeCloud); //当前原始点云 laserCloudAllLast
      for (int i = 0; i < cloudQueue.size(); ++i)
      {
        *cureKeyframeCloud += cloudQueue[i];
      }
      pcl::copyPointCloud(*laserCloudAllLast, *cureKeyframeCloudTmp);
      // pcl::copyPointCloud(*laserCloudAllLastDS, *cureKeyframeCloudTmp);
      // 进行到绝对坐标的变换 使用transformTobeMapped
      // 一个疑问 这里使用transformTobeMapp来进行变换 但我们的局部地图是用cloudKeyPoses6D->points[thisK变换来的
      // 这会使得这两点云不在同一坐标系，而提取局部地图时不能使用transform来变换，所以我们把下面这个改成也用cloudKeyPose来变换?
      // 但是当前帧点云对应的cloudKeyPose6d还不存在呢，需要我们优化后来赋值，可以选择使用上一帧的，但和使用transformTobeMapped也差不多
      // 只不过， 如果imuavil和odoavil的flag都不可用的话，会使得transForm的 3 4 5 没有值 但现在看上去有值 所以我们不改变这个
      cureKeyframeCloudTmp = transformPointClouduseTransformTobeMapped(cureKeyframeCloudTmp);
      *cureKeyframeCloud += *cureKeyframeCloudTmp;
      // cureKeyframeCloud = transformPointCloud(cureKeyframeCloud,  &cloudKeyPoses6D->back());
      // 获取局部地图点云
      // 注意 这个点云是使用cloudKeyPose6D来变换到绝对 坐标系的
      pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
      pcl::copyPointCloud(*laserCloudAllFromMap, *prevKeyframeCloud);
      // pcl::copyPointCloud(*laserCloudAllFromMapDS, *prevKeyframeCloud); //changed to dowmsample

      /***************now start ndt method***********************/
      rclcpp::Clock clock;
      rclcpp::Time begin2 = clock.now();
      Eigen::Affine3f correctionLidarFrame;
      if (ndt_search_method == 0) // for kdtree search
      {
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(ndtEpsion); // 0.01
        ndt.setStepSize(ndtStepSize);
        ndt.setResolution(ndtResolution);
        ndt.setRANSACIterations(5);
        ndt.setRANSACOutlierRejectionThreshold(0.05); // default 0.05m
        ndt.setMaximumIterations(ndtMaxInter);
        ndt.setInputCloud(cureKeyframeCloud);
        ndt.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        rclcpp::Time start = clock.now();

        ndt.align(*unused_result);
        ofs << stamp2Sec(clock.now(), start) << ",";
        //  get ndt iteration num
        // ofs << ndt.getFinalNumIteration() << "," <<  ndt.getTransformationProbability() << "," << ndt.getFitnessScore() << ",";
        ofs << ndt.getFinalNumIteration() << "," << ndt.getFitnessScore() << std::endl;
        RCLCPP_DEBUG(get_logger(), "iter:%d elapsed_time:%.8f fit_score:%.4f", ndt.getFinalNumIteration(),
                     stamp2Sec(clock.now(), start), ndt.getFitnessScore());
        if (!ndt.hasConverged() && ndt.getFitnessScore() > historyKeyframeFitnessScore)
        {
          RCLCPP_ERROR(get_logger(), "ndt match failure");
          correctionLidarFrame = Eigen::Matrix4f::Identity();
        }
        else
        {
          correctionLidarFrame = ndt.getFinalTransformation();
        }
        std::cout << "direct call pcl totoal use:" << stamp2Sec(clock.now(), begin2) * 1000 << "[ms]" << std::endl;
        if (ndt.getFinalNumIteration() > ndtMaxInter || stamp2Sec(clock.now(), start) > 5 * 0.1)
        {
          RCLCPP_ERROR(get_logger(), "front_end registration may in stuck!");
          correctionLidarFrame = Eigen::Matrix4f::Identity();
        }
      }
      /*********now start ndt_omp***********/
      else
      {
        rclcpp::Time begin = clock.now();
        pclomp::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(ndtEpsion); // 0.01
        ndt.setStepSize(ndtStepSize);
        ndt.setResolution(ndtResolution);
        ndt.setNumThreads(10);
        if (ndt_search_method == 1) // for direct7
          ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
        else if (ndt_search_method == 2) // for direct1
          ndt.setNeighborhoodSearchMethod(pclomp::DIRECT1);
        else // 默认用Direct7
          ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);
        ndt.setMaximumIterations(ndtMaxInter);
        ndt.setInputCloud(cureKeyframeCloud);
        ndt.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        rclcpp::Time start = clock.now();
        ndt.align(*unused_result);
        ofs << stamp2Sec(clock.now(), start) << ",";
        //  get ndt iteration num
        ofs << ndt.getFinalNumIteration() << "," << ndt.getFitnessScore() << std::endl;

        RCLCPP_DEBUG(get_logger(), "iter:%d elapsed time:%.8f probability:%f fitnessScore:%f", ndt.getFinalNumIteration(),
                     stamp2Sec(clock.now(), start), ndt.getTransformationProbability(), ndt.getFitnessScore());
        if (!ndt.hasConverged() && ndt.getFitnessScore() > historyKeyframeFitnessScore)
        {
          RCLCPP_INFO(get_logger(), "\033[1;34m---->ndt match failure!\033[0m");
          correctionLidarFrame = Eigen::Matrix4f::Identity();
        }
        else
        {
          correctionLidarFrame = ndt.getFinalTransformation();

          // 拿到ndt的hessian矩阵对其进行svd分解 TODO：用于退化检测
          Eigen::Matrix<double, 6, 6> ndtHessian;
          ndt.getHessian(ndtHessian);
          Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(ndtHessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
          Eigen::Matrix<double, 6, 1> singularValues = svd.singularValues();
          // std::cout << "Singular Values:" << std::endl;
          // std::cout << singularValues << std::endl;
          double maxVal = singularValues.maxCoeff();
          double minVal = singularValues.minCoeff();
          if (minVal > 0.0000001 || minVal < -0.0000001)
          {
            double ratio = maxVal / minVal;
            // std ::cout << "maxVal:" << maxVal << " minVal:" << minVal << " ratio:" << ratio << std::endl;
          }

          // 将ratio保存为txt文件
        }
        // std::cout << "direct call totoal use:" << (ros::WallTime::now() - begin1).toSec() * 1000 << "[ms]" << std::endl;
        //  add by zhaoyz 2022.11.23 这里初步根据ndt的匹配次数和时间实现一个粗略的不稳定(退化)匹配的检测
        if (ndt.getFinalNumIteration() > 0.3 * ndtMaxInter || stamp2Sec(clock.now(), start) > 5 * 0.1)
        {
          RCLCPP_ERROR(get_logger(), "front_end registration may in stuck!");
          //@todo 不稳定处理 拿imu的位置/做一次滤波来替代correctionLidarFrame;
        }
      }
      float x, y, z, roll, pitch, yaw;
      Eigen::Affine3f tBefore = trans2Affine3f(transformTobeMapped); // transformTobeMapped这里相当于已经是经过initial guess计算过了
      // transform from world origin to corrected pose
      Eigen::Affine3f tAfter = correctionLidarFrame * tBefore;
      // 从中获得欧拉角和变换等
      pcl::getTranslationAndEulerAngles(tAfter, x, y, z, roll, pitch, yaw);
      //std::cout << x << " " << y << " " << z << " " << roll << " " << pitch << " " << yaw << std::endl;
      // 然后在变换回trans，用于下面的transformUpdate
      transformTobeMapped[0] = roll;
      transformTobeMapped[1] = pitch;
      transformTobeMapped[2] = yaw;
      transformTobeMapped[3] = x;
      transformTobeMapped[4] = y;
      transformTobeMapped[5] = z;
      static bool first_frame = true;
      static double initial_z = 0.0;
      if (is_use_gps_a)
      {
        if (first_frame)
        {
          transformTobeMapped[5] = z;
          initial_z = z;
          first_frame = false;
        }
        geometry_msgs::msg::PointStamped curr_gps;
        if (getCurrGPSPoint(timeLaserInfoCur, curr_gps))
          transformTobeMapped[5] = curr_gps.point.z;
        gps_convert_z = curr_gps.point.z;
        // transformTobeMapped[5] = curr_gps_z - last_gps_z;
        // gps_z_value.push_back(curr_gps.point.z);
        RCLCPP_WARN(get_logger(), "z--->est:%f gt_1:%f gt_2:%f", z, curr_gps_z - last_gps_z, curr_gps.point.z);
        last_gps_z = curr_gps_z;
      }

      transformUpdate();
      cloudQueue.push_back(*transformPointClouduseTransformTobeMapped(laserCloudAllLast));
      while (cloudQueue.size() > multi_frame)
      {
        cloudQueue.pop_front();
      }
    }
    else
    {
      RCLCPP_WARN(get_logger(), "Not enough Points! ");
    }
  }

  // 对transformTobeMapped和imu的roll pitch角度进行加权平均处理
  void transformUpdate()
  {
    if (cloudInfo.imu_available == true)
    {
      if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
      {
        double imuWeight = imuRPYWeight;
        tf2::Quaternion imuQuaternion;
        tf2::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp roll
        transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
        imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
        tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[0] = rollMid;

        // slerp pitch
        transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
        imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
        tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
        transformTobeMapped[1] = pitchMid;
      }
    }

    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
  }

  float constraintTransformation(float value, float limit)
  {
    if (value < -limit)
      value = -limit;
    if (value > limit)
      value = limit;

    return value;
  }

  bool saveFrame()
  {
    if (cloudKeyPoses3D->points.empty() || mMcloudKeyPoses3D->points.empty())
    {
      std::cout << "[saveFrame] first frame--->return" << std::endl;
      return true;
    }

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4],
                                                        transformTobeMapped[5], transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
    // 这里判断是否满足插入关键帧的条件,即角度大于0.2rad 距离大于1.0m
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
      return false;

    return true;
  }

  void addOdomFactor()
  {
    if (cloudKeyPoses3D->points.empty())
    {

      std::cout << "first add odometry factor!"
                << "firstkey:" << firstKey << std::endl;

      // 第一帧时，初始化gtsam参数
      noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e-8, 1e-8, 1e-8).finished()); // rad*rad, meter*meter
      // 先验 0是key 第二个参数是先验位姿，最后一个参数是噪声模型，如果没看错的话，先验应该默认初始化成0了
      gtSAMgraph->add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
      // 添加节点的初始估计值
      initialEstimate->insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else if(cloudKeyPoses3D->points.size() == firstKey && mapping_mode==2)
    {
      std::cout << "first add odometry factor!"
                << "firstkey:" << firstKey << std::endl;

      // 第一帧时，初始化gtsam参数
      noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) <<  1e8, 1e8, 1e8, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
      // 先验 0是key 第二个参数是先验位姿，最后一个参数是噪声模型，如果没看错的话，先验应该默认初始化成0了
      gtSAMgraph->add(PriorFactor<Pose3>(firstKey, trans2gtsamPose(transformTobeMapped), priorNoise));
      // 添加节点的初始估计值
      initialEstimate->insert(firstKey, trans2gtsamPose(transformTobeMapped));      
    }
    else
    {
      // 之后是添加二元的因子
      // noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      // 从这里可以看出，poseFrom不谈，poseTo一定是绝对坐标系下的
      // 添加的Factor值是 poseFrom.between(poseTo)    = poseFrom.inverse * poseTo;
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
      gtsam::Pose3 relPose = poseFrom.between(poseTo);

      poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
      poseTo = trans2gtsamPose(transformTobeMapped);
      relPose = poseFrom.between(poseTo);
      noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      gtSAMgraph->add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), relPose, odometryNoise));

      initialEstimate->insert(cloudKeyPoses3D->size(), poseTo);
    }
  }

  void addGPSFactor()
  {
    if (gpsQueue.empty())
      return;

    // wait for system initialized and settles down
    // 系统初始化且位移一段时间了再考虑要不要加gpsfactor
    if (cloudKeyPoses3D->points.empty())
      return;
    else
    {
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < gpsDisctanceThreshold)
        {
            return;        
        }
        if(mapping_mode == 2 && mMcloudKeyPoses3D->size() < 20)
        {
            return;
        }
    }

    // RCLCPP_DEBUG(get_logger(), "poseCov: rx:%f ry:%f", poseCovariance(3, 3), poseCovariance(4, 4));
    // if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
    //   return;
    // last gps position
    static PointType lastGPSPoint;

    while (!gpsQueue.empty())
    {
      if (stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 1.2) // 0.2
      {
        // message too old
        gpsQueue.pop_front();
      }
      else if (stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 1.2) // 0.2
      {
        // message too new
        break;
      }
      else // 找到timeLaserInfoCur前后0.2s内的gps数据, 这里的0.2应该根据实际gps的频率来定。
      {
        nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
        gpsQueue.pop_front();

        // GPS too noisy, skip 23.3.3目对于rs的这里是固定的0.1
        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];
        // 目前这里的noise_x,noise_y,noise_z都是0,因为gps的原始消息里也是0
        // 所以需要给gps的协方差字段赋值，这里相当于百分百的相信gps消息了
        // gpsCovThreshold目前设置的值为2.0
        // std::cout << "gps noise:" << noise_x << " " << noise_y << " covThreshold:" << gpsCovThreshold;
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) // 0.135335 0.2
          continue;

        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        if (!useGpsElevation)
        {
          // gps 的z一般不可信 用radar odometry估计出来的z值
        //   gps_z = transformTobeMapped[5];
          gps_z = 0.1;
          noise_z = 0.01;
        }

        // GPS not properly initialized (0,0,0)
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
          continue;

        // Add GPS every a few meters
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (pointDistance(curGPSPoint, lastGPSPoint) < gpsDisctanceThreshold) // 至少间隔5m以上 1.0
          continue;
        else
          lastGPSPoint = curGPSPoint;

        gtsam::Vector Vector3(3);
        RCLCPP_INFO(get_logger(), "\033[1;32m---->add one gps factor : %d,%d,%d \033[0m", gps_x, gps_y, gps_z);
        // std::cout << " add one gps factor." << gps_x << " " << gps_y << " " << gps_z << std::endl;
        // 意味着添加的gps的因子的噪声至少为1.0
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 0.01f);
        // Vector3 << 2.0 , 2.0, 2.0;
        noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
        gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
        gtSAMgraph->add(gps_factor);
        aLoopIsClosed = true;

        break;
      }
    }
  }

  void addPlanarFactor()
  {
    if(cloudKeyPoses3D->points.size() == firstKey)
    {
        return;
    }

    static int add_planar_factor_count = 0;
    if(add_planar_factor_count % 1 == 0)
    {
        double measuredZValue = 0.1; // 期望的 Z 值
        // 构造噪声
        auto noise = noiseModel::Diagonal::Sigmas(Vector1(1e-4));
        // 构造因子
        gtSAMgraph->add((PlanarFactor(cloudKeyPoses3D->size(), measuredZValue, noise)));
        std::cout << " add one planar factor." << std::endl;
    }
    add_planar_factor_count++;
  }

  void addGPSZFactor()
  {
    // wait for system initialized and settles down
    // 系统初始化且位移一段时间了再考虑要不要加gpsfactor
    if (cloudKeyPoses3D->points.empty())
      return;
    float noise_x = 1e-0;
    float noise_y = 1e-0;
    float noise_z = 1e-6; // 相信gps的z值 这里的参数需要调试，不过在odometry factor中就是这个值
    geometry_msgs::msg::PointStamped curr_gps_point;
    if (!getCurrGPSPoint(timeLaserInfoCur, curr_gps_point))
    {
      RCLCPP_ERROR(get_logger(), "Get GPS Point failure!");
      return;
    }

    float gps_x = transformTobeMapped[3];
    float gps_y = transformTobeMapped[4];
    float gps_z = curr_gps_point.point.z;
    gps_convert_z = curr_gps_point.point.z;
    std::cout << "add sim gps factor:" << transformTobeMapped[3] << " " << transformTobeMapped[4] << std::endl;

    gtsam::Vector Vector3(3);

    // 意味着添加的gps的因子的噪声至少为1.0
    Vector3 << noise_x, noise_y, noise_z;
    noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
    gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
    gtSAMgraph->add(gps_factor);
  }

  void addLoopFactor()
  {
    if (loopIndexQueue.empty())
      return;

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
      int indexFrom = loopIndexQueue[i].first;
      int indexTo = loopIndexQueue[i].second;
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      auto noiseBetween = loopNoiseQueue[i]; // giseop for polymorhpism // shared_ptr<gtsam::noiseModel::Base>, typedef noiseModel::Base::shared_ptr gtsam::SharedNoiseModel
      gtSAMgraph->add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
    aLoopIsClosedOnlyICP = true;
  }

  bool saveMap(std::string save_dir, float resolution)
  {
    string saveMapDirectory;
    saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    cout << "Save destination: " << saveMapDirectory << endl;
    // create directory and remove old files;
    int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
    unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
    // save key frame transformations
    // pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
    pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
    for (int i = 0; i < (*cloudKeyPoses6D).size(); ++i)
    {
        save_trajectory.open(saveMapDirectory+"/trajectory.traj", std::ofstream::out | std::ofstream::app);
        save_trajectory.precision(18);
        double roll = (*cloudKeyPoses6D).points[i].roll;
        double pitch = (*cloudKeyPoses6D).points[i].pitch;
        double yaw = (*cloudKeyPoses6D).points[i].yaw;
        tf2::Quaternion quattf;
        quattf.setRPY(roll, pitch, yaw);

        geometry_msgs::msg::Quaternion quat;
        quat.x = quattf.x();
        quat.y = quattf.y();
        quat.z = quattf.z();
        quat.w = quattf.w();
        save_trajectory << keyFramePoseTimestamp[i] << "," << (*cloudKeyPoses6D).points[i].x  << "," << (*cloudKeyPoses6D).points[i].y << "," << 0 << ","
                                            <<  quat.x  << "," << quat.y << "," << quat.z << "," << quat.w << "," << vcu_gear[i]
                                            << std::endl;
        save_trajectory.close();
    }

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

    if (is_filter_cloud_map_pcd)
    {
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
    }
    ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);

    // 地图压缩
    // 将XYZI->XYZ
    if (map_compression)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr globalMapA(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::copyPointCloud(*globalMapCloud, *globalMapA);

      bool showStatistics = true;
      pcl::io::compression_Profiles_e compressionProfile = pcl::io::MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;

      pcl::io::OctreePointCloudCompression<pcl::PointXYZ> *PointCloudEncoder;
      PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compressionProfile, showStatistics); // 输入参数
      std::stringstream compressedData;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>());
      rclcpp::Clock clock;
      rclcpp::Time start = clock.now();
      PointCloudEncoder->encodePointCloud(globalMapA, compressedData);
      RCLCPP_WARN(get_logger(), "point Cloud compression esclaped: %.8f", stamp2Sec(clock.now(), start));
      // 保存为二进制的格式
      ofstream OutbitstreamFile(saveMapDirectory + "/GlobalMapCompressed.bin", fstream::binary | fstream::out);
      OutbitstreamFile << compressedData.str();

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n"
           << endl;
    }

    // 保存初始时刻的gps
    std::ofstream save_gps_pose_ofs;
    save_gps_pose_ofs.open(saveMapDirectory + "/gps_keyframes_poses.gp", std::ios::out | std::ios::app);
    if (!save_gps_pose_ofs.is_open())
      RCLCPP_ERROR(get_logger(), "%s can't open!", (saveMapDirectory + "/gps_keyframes_poses.gp").c_str());
    save_gps_pose_ofs << "#gps pose align to map, keyframe_ind latitude longitude altitude." << std::endl;
    for (auto gps : gps_key_map)
      save_gps_pose_ofs << gps.first << " " << std::setprecision(12) << gps.second.at(0) << " " << gps.second.at(1) << " " << gps.second.at(2) << std::endl;
    save_gps_pose_ofs.close();

    if (is_save_tum_pose)
    {
      for (int i = 0; i < (*cloudKeyPoses6D).size(); ++i)
      {
        save_tum_pose.open(saveMapDirectory+"/trajectory_mapping.txt", std::ofstream::out | std::ofstream::app);
        save_tum_pose.precision(18);
        save_tum_pose << keyFramePoseTimestamp[i] << " "
                      << (*cloudKeyPoses6D).points[i].x << " " << (*cloudKeyPoses6D).points[i].y << " " << 0.0 << " ";
        double roll = (*cloudKeyPoses6D).points[i].roll;
        double pitch = (*cloudKeyPoses6D).points[i].pitch;
        double yaw = (*cloudKeyPoses6D).points[i].yaw;
        tf2::Quaternion quat1;
        quat1.setRPY(roll, pitch, yaw);

        // 将tf2::Quaternion转换为geometry_msgs::Quaternion
        geometry_msgs::msg::Quaternion quat;
        tf2::convert(quat1, quat);
        // geometry_msgs::Quaternion quat = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        // save_tum_pose << quat.x << " " << quat.y << " " << quat.z << " " << quat.w << std::endl;
        save_tum_pose << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
        save_tum_pose.close();
      }
    }

    if (is_visualization_graph)
    {
      gtSAMgraph->saveGraph(graph_viz, *isamCurrentEstimate);
      graph_viz.close();
    }

    if (mapping_mode == 2) // remember mode
    {
      std::cout << "mapping mode is remember mode." << std::endl;
      saveGraph(saveMapDirectory);
      saveKeyframePoints(saveMapDirectory);
      // printSaveInfo();
    }

    return true;
  }

  void saveGraph(const std::string &save_dir)
  {
    // 调用dataset.h中的writeG2o()接口.这个接口中只实现了保存between factor.没有实现其他的factor的保存
    // 再创建一层目录
    std::string saveMapDirectory = save_dir + "/graph";
    int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
    unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
    std::cout << "gtSAMgraph size:" << gtSAMgraph->size() << std::endl;
    writeG2o(*gtSAMgraph, *isamCurrentEstimate, saveMapDirectory + "/slam_graph.txt");
  }

  void saveKeyframePoints(const std::string &save_dir)
  {
    // 再创建一层目录
    std::string saveMapDirectory = save_dir + "/all_clouds";
    int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
    unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());

    pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
    {
      // 这里只需要保存原始base下的点云，变得是pose,与点云无关。
      //*globalMapCloud += *transformPointCloud(allCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/" + std::to_string(i) + ".pcd", *allCloudKeyFrames[i]);
    }
  }

  void closeMappingNode()
  {
    for (auto node : mapping_nodes)
    {
      int pid = getPidImpl(node);
      if (killByPid(pid))
      {
        std::cout << "kill " << node << " success" << std::endl;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
  }

  void saveMapping()
  {
    RCLCPP_WARN(get_logger(), "Maybe mapping can finished!");
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
      RCLCPP_ERROR(get_logger(), "in 5s, save map failure, please check your map_dir config or call save map by manual!");
    }
    else
    {
      RCLCPP_INFO(get_logger(), "save map succeed!");
      // 向定位节点发布建图结束的命令 连续发布10帧
      for (int i = 0; i < 10; ++i)
      {
        std_msgs::msg::Empty msg;
        notisfy_local_pub->publish(msg);
        rclcpp::sleep_for(std::chrono::milliseconds(200));
      }
    }
    // 关掉建图节点
    // closeMappingNode();
  }

  bool autoEndingMappingMethod0(const PointType &KeyframesPose3D)
  {
    // 利用关键帧位置+时间戳判断圈数
    // 时间戳初始化 保存第一个关键帧的时间与位姿 将其作为起点
    if (firstLoop)
    {
      startTime = timeLaserInfoCur;
      cloudLoopDis3D->push_back(KeyframesPose3D);
      firstLoop = false;
    }
    endTime = timeLaserInfoCur;
    cloudLoopDis3D->push_back(KeyframesPose3D);
    double dt = endTime - startTime;
    minDis = pointDistance(cloudLoopDis3D->front(), cloudLoopDis3D->back());
    // 由于第一圈回环位置的不确定性，需要特殊对待
    if (aLoopIsClosedOnlyICP)
    {
      // 回环约束次数
      loopNums++;
      if (dt > start_end_time)
      {
        if (driveRounds == 0)
        {
          if (minDis < start_end_dis + 100)
          {
            // 圈数加一
            driveRounds++;
            // 重新给起点时间/空间变量赋值
            cloudLoopDis3D->clear();
            startTime = timeLaserInfoCur;
            cloudLoopDis3D->push_back(KeyframesPose3D);
          }
        }
        else
        {
          if (minDis < start_end_dis)
          {
            // 圈数加一
            driveRounds++;
            // 重新给起点时间/空间变量赋值
            cloudLoopDis3D->clear();
            startTime = timeLaserInfoCur;
            cloudLoopDis3D->push_back(KeyframesPose3D);
          }
        }
      }
    }
    RCLCPP_INFO(get_logger(), "vehicle drive one round, has drived %d", driveRounds);
    RCLCPP_INFO(get_logger(), "loopNums %d", loopNums);
    // 建图结束条件
    if (driveRounds > drive_rounds)
    {
      if (loopNums > loop_nums)
      {
        return true;
      }
      return false;
    }
  }

  bool autoEndingMappingMethod1()
  {
    // 计算回环因子图优化次数
    if (aLoopIsClosedOnlyICP)
    {
      loopNums++;
    }
    RCLCPP_DEBUG(get_logger(), "loopNums %d", loopNums);
    // 回环约束30次则结束建图
    if (loopNums > loop_nums)
    {
      return true;
    }
    return false;
  }

  bool autoEndingMappingMethod2()
  {
    if (gpsLoop3D->size() > 1)
    {
      if (firstLoop)
      {
        startTime = gpsMsgTime;
        firstLoop = false;
      }
      endTime = gpsMsgTime;
      minDis = pointDistance(gpsLoop3D->front(), gpsLoop3D->back());
      if (minDis < start_end_dis)
      {
        if (double(endTime - startTime) > start_end_time)
        {
          driveRounds++;
          startTime = gpsMsgTime;
          gpsLoop3D->clear();
        }
      }
      RCLCPP_DEBUG(get_logger(), "driveRounds %d", driveRounds);
    }
    // 两圈以上则结束建图
    if (driveRounds > drive_rounds)
    {
      return true;
    }
    return false;
  }

  bool autoEndingMappingMethod3()
  {
    if (gpsLoop3D->size() > 0)
    {
      if (firstLoop)
      {
        startTime = gpsMsgTime;
        firstLoop = false;
      }
      endTime = gpsMsgTime;
      minDis = pointDistance(gpsLoop3D->front(), gpsLoop3D->back());
      RCLCPP_DEBUG(get_logger(), "vehicle drive one round, has drived %d", driveRounds);
      if (minDis < start_end_dis)
      {
        if (double(endTime - startTime) > start_end_time)
        {
          driveRounds++;
          startTime = gpsMsgTime;
          gpsLoop3D->clear();
        }
      }
    }
    // 计算回环因子图优化次数
    if (aLoopIsClosedOnlyICP)
    {
      loopNums++;
    }
    // 两圈以上并且回环约束30次则结束建图
    RCLCPP_DEBUG(get_logger(), "driveRounds %d", driveRounds);
    RCLCPP_DEBUG(get_logger(), "loopNums %d", loopNums);
    if (driveRounds > drive_rounds)
    {
      if (loopNums > loop_nums)
      {
        return true;
      }
    }
    return false;
  }

  bool endingMapping(const PointType &KeyframesPose3D)
  {
    //========方案1 按照以及是否产生回环判断检测走过的圈数,并按照圈数以及回环次数进行判断并结束建图=========
    if (auto_ending_mapping_method == 0)
    {
      return autoEndingMappingMethod0(KeyframesPose3D);
    }
    //========方案2：通过判断回环检测次数结束建图========
    if (auto_ending_mapping_method == 1)
    {
      return autoEndingMappingMethod1();
    }
    //========方案3：通过gps判断走过的圈数结束建图========
    if (auto_ending_mapping_method == 2)
    {
      return autoEndingMappingMethod2();
    }
    //========方案4：通过gps判断走过的圈数+回环次数结束建图========
    if (auto_ending_mapping_method == 3)
    {
      return autoEndingMappingMethod3();
    }
  }

  void closeNodesSub(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "close nodes sub...");
    closeMappingNode();
  }

  void saveKeyFramesAndFactor()
  {
    // 判断是否经过了一段距离 其实就是提关键帧
    if (saveFrame() == false)
      return;
    

    // odom factor
    addOdomFactor(); // radar odometry的结果

    if (is_use_gps_c)
      addGPSZFactor();

    // gps factor
    addGPSFactor();

    // loop factor
    addLoopFactor(); // 闭环检测结果

    // Planar constraint factor
    // addPlanarFactor();
    // update iSAM
    // gtSAMgraph->print();
    // initialEstimate.print();

    // save key poses
    // 优化完成后，保存3D和6D位姿
    PointType thisPose3D;
    PointTypePose thisPose6D;
    Pose3 latestEstimate;
    if (is_global_optimization)
    {
      gtsam::LevenbergMarquardtParams lmparameters;
      lmparameters.relativeErrorTol = 1e-5;
      lmparameters.maxIterations = 100; // 这里的最大迭代次数可能需要时间的消耗调整
      gtsam::LevenbergMarquardtOptimizer optimizer(*gtSAMgraph, *initialEstimate, lmparameters);
      *isamCurrentEstimate = optimizer.optimize();
    
    
    }
    else
    {
      isam->update(*gtSAMgraph, *initialEstimate);
      isam->update();

      // 这里的逻辑是，当检测到闭环后，更新因子图；
      // 这里只将最后一个pose的最新的值拿出来；
      // 在correctPose()函数里将之前所有的keyPose更新一次
      // 二者并不冲突. 2023.1.6
      if (aLoopIsClosed == true)
      {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
      }

      gtSAMgraph->resize(0);
      initialEstimate->clear();

      *isamCurrentEstimate = isam->calculateEstimate();
      poseCovariance = isam->marginalCovariance(isamCurrentEstimate->size() - 1);
    }

    latestEstimate = isamCurrentEstimate->at<Pose3>(isamCurrentEstimate->size() - 1);

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();

    // std::cout << "est:" << latestEstimate.translation().x() << " " << latestEstimate.translation().y() << " "
    //           << latestEstimate.translation().z() << std::endl;
    static bool is_initial = true;
    static double initial_z = 0.0;
    if (is_use_gps_z)
    {
      if (is_initial)
      {
        thisPose3D.z = latestEstimate.translation().z();
        initial_z = thisPose3D.z;
      }
      else
        thisPose3D.z = initial_z + curr_gps_z - last_gps_z;
      last_gps_z = curr_gps_z;
    }

    else
      thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index、
    // 这里终于push_back cloudKeyPoses3D
    cloudKeyPoses3D->push_back(thisPose3D);
    mMcloudKeyPoses3D->push_back(thisPose3D);
    keyFramePoseTimestamp.push_back(stamp2Sec(timeLaserInfoStamp));

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = timeLaserInfoCur;
    cloudKeyPoses6D->push_back(thisPose6D);
    mMcloudKeyPoses6D->push_back(thisPose6D);

    // 将新的关键帧位姿存入mappingPath[1]，后续如果更新，则同步更新存入mappingPath的内容
    mappingPath[1] = *mMcloudKeyPoses6D;

    // 寻找对应当前时刻的Vcu_Data的档位
    int gear = associateKeyframeWithVcuDAata(timeLaserInfoCur);
    vcu_gear.push_back(gear);
    // std::cout << "gear: " << gear << std::endl;

    // 这里打印一下poseCovariance的大小，应该是6*6
    // RCLCPP_INFO_ONCE(get_logger(), "poseCovaraince is matrix: %d * %d", poseCovariance.rows(), poseCovariance.cols());
    // 这里再判断一下协方差的大小？ 这里需要把x y yaw的协方差打印出来，看下在 闭环检测后优化的协方差大小
    // 只有当闭环后对误差的校正比较显著时，才进行闭环统计？（协方差的考量后续再做处理）

    // save updated transform
    transformTobeMapped[0] = latestEstimate.rotation().roll();
    transformTobeMapped[1] = latestEstimate.rotation().pitch();
    transformTobeMapped[2] = latestEstimate.rotation().yaw();
    transformTobeMapped[3] = latestEstimate.translation().x();
    transformTobeMapped[4] = latestEstimate.translation().y();
    transformTobeMapped[5] = latestEstimate.translation().z();

    // 将所有帧的gps的数据保存下来
    while (!gpsRawQueue.empty())
    {
      // 这里应该有个处理过程的时间，不过可以忽略
      if (stamp2Sec(gpsRawQueue.front().header.stamp) < timeLaserInfoCur - 1.01)
        gpsRawQueue.pop_front(); // 如果imu队列里的front时间比timeScanCur提前太多，就pop掉，直到接近
      else
        break;
    }
    if (gpsRawQueue.empty())
    {
      // RCLCPP_INFO(get_logger(), "GPS is not available!");
      // 要不要考虑等待呢
    }
    else
    {
      sensor_msgs::msg::NavSatFix gps_msg = gpsRawQueue.front();
      // gps_lat = gps_msg.latitude;
      // gps_long = gps_msg.longitude;
      // gps_alt = gps_msg.altitude;

      // if (gps_msg.status.status != 69 && gps_msg.status.status != 75 && gps_msg.status.status != 85 && gps_msg.status.status != 91)
      // {
      //     RCLCPP_ERROR(get_logger(), "gps staus %d is not availble.", gps_msg.status.status);
      //     // return;
      // }
      // else
      // {
      std::vector<double> gps_raw_vec;
      gps_raw_vec.push_back(gps_msg.latitude);
      gps_raw_vec.push_back(gps_msg.longitude);
      gps_raw_vec.push_back(gps_msg.altitude);
      gps_key_map.insert(std::pair<int, std::vector<double>>(cloudKeyPoses6D->size() - 1, gps_raw_vec));
      // }
    }

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisAllKeyFrame(new pcl::PointCloud<PointType>());

    // changed by zhaoyz 保存不降采样的点云
    pcl::copyPointCloud(*laserCloudAllLast, *thisAllKeyFrame);

    allCloudKeyFrames.push_back(thisAllKeyFrame);

    //========保存地图========
    if (is_auto_save_map)
    {
      if (endingMapping(thisPose3D)) // 存放具体方案的函数
      {
        MappingStatus currentStatus = EndMapping;
        publishMappingStatus(currentStatus, 5); // 10
        saveMapping();
      }
    }
    // std::cout << "updateOldPath(thisPose6D)" << std::endl;
    updateNewPath(thisPose6D);
  }

  void correctPoses()
  {
    if (cloudKeyPoses3D->points.empty())
      return;

    isLoopClosed = aLoopIsClosed;
    if (aLoopIsClosed == true)
    {
      // clear map cache
      laserCloudMapContainer.clear();


      // clear path
      globalOldPath.poses.clear();
      globalNewPath.poses.clear();
      // update key poses
      // 这里直接拿更新后因子图的值
      int numPoses = firstKey;
      for (int i = 0; i < numPoses; ++i)
      {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate->at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate->at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate->at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll = isamCurrentEstimate->at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate->at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate->at<Pose3>(i).rotation().yaw();

        updateOldPath(cloudKeyPoses6D->points[i]);
      }

      for(int i = firstKey; i < isamCurrentEstimate->size(); ++i)
      {
        cloudKeyPoses3D->points[i].x = isamCurrentEstimate->at<Pose3>(i).translation().x();
        cloudKeyPoses3D->points[i].y = isamCurrentEstimate->at<Pose3>(i).translation().y();
        cloudKeyPoses3D->points[i].z = isamCurrentEstimate->at<Pose3>(i).translation().z();

        cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
        cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
        cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
        cloudKeyPoses6D->points[i].roll = isamCurrentEstimate->at<Pose3>(i).rotation().roll();
        cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate->at<Pose3>(i).rotation().pitch();
        cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate->at<Pose3>(i).rotation().yaw();

        updateNewPath(cloudKeyPoses6D->points[i]);
      }

      aLoopIsClosed = false;
    }

    // 再次更新incrementalOdometryAffineBack这个变量，看是否有问题
    // incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
  }

  void updateOldPath(const PointTypePose &pose_in)
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf2::Quaternion q;
    q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalOldPath.poses.push_back(pose_stamped);
  }

  void updateNewPath(const PointTypePose &pose_in)
  {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf2::Quaternion q;
    q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalNewPath.poses.push_back(pose_stamped);
  }

  void publishOdometry()
  {
    // Publish odometry for ROS (global)
    nav_msgs::msg::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = timeLaserInfoStamp;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
    tf2::Quaternion quat_tf;
    quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    geometry_msgs::msg::Quaternion quat_msg;
    tf2::convert(quat_tf, quat_msg);
    laserOdometryROS.pose.pose.orientation = quat_msg;
    pubLaserOdometryGlobal->publish(laserOdometryROS);

    // Publish TF
    quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
    tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, odometryFrame);
    geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
    tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
    trans_odom_to_lidar.child_frame_id = "radar_link";
    br->sendTransform(trans_odom_to_lidar);

    // Publish odometry for ROS (incremental)
    static bool lastIncreOdomPubFlag = false;
    static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3f increOdomAffine;
    if (lastIncreOdomPubFlag == false)
    {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = trans2Affine3f(transformTobeMapped);
    }
    else
    {
      // 这里计算odometry的增量，是在scan2MapOptimization中获取的，而不是整个图优化后获取
      // 对于匹配成功的情况下，在scan2MapOptimization中获取问题不大
      // 但对于匹配失败的情况，应该在图优化后再次更新这个值
      // 这里的odometry_incremental的求解不是很明白
      // 23.5.24这里的affineIncre是匹配前后的变化，即匹配对预积分结果的修正量，increodomAffine其实还是绝对量
      // 这里所说的incremental实际说的是把匹配处理后的结果即前后帧的相对变换作用到上一帧的匹配结果上了。
      // 实际上这里发布的laserOdomIncremental和laserOdometryROS在position上是一样的，小的差别是多做了roll pitch 的处理而已。
      Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
      increOdomAffine = increOdomAffine * affineIncre;
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
      if (cloudInfo.imu_available == true)
      {
        if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
        {
          double imuWeight = 0.1;
          tf2::Quaternion imuQuaternion;
          tf2::Quaternion transformQuaternion;
          double rollMid, pitchMid, yawMid;

          // slerp roll 四元数球面线性插值，原始imu预积分出来的，权重只占0.1,更加相信匹配出来的roll和pitch
          transformQuaternion.setRPY(roll, 0, 0);
          imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
          tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          roll = rollMid;

          // slerp pitch
          transformQuaternion.setRPY(0, pitch, 0);
          imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
          tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
          pitch = pitchMid;
        }
      }
      laserOdomIncremental.header.stamp = timeLaserInfoStamp;
      laserOdomIncremental.header.frame_id = odometryFrame;
      laserOdomIncremental.child_frame_id = "odom_mapping";
      laserOdomIncremental.pose.pose.position.x = x;
      laserOdomIncremental.pose.pose.position.y = y;
      laserOdomIncremental.pose.pose.position.z = z;
      tf2::Quaternion quat_tf;
      quat_tf.setRPY(roll, pitch, yaw);
      geometry_msgs::msg::Quaternion quat_msg;
      tf2::convert(quat_tf, quat_msg);
      laserOdomIncremental.pose.pose.orientation = quat_msg;
      if (isDegenerate)
        laserOdomIncremental.pose.covariance[0] = 1;
      else
        laserOdomIncremental.pose.covariance[0] = 0;
    }
    pubLaserOdometryIncremental->publish(laserOdomIncremental);
  }

  void publishFrames()
  {
    if (cloudKeyPoses3D->points.empty())
      return;
    // publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
    // Publish surrounding key frames
    // publishCloud(pubRecentKeyFrames, laserCloudAllFromMapDS, timeLaserInfoStamp, odometryFrame);
    // for submap test changed to laserCloudAllFromMap
    publishCloud(pubRecentKeyFrames, laserCloudAllFromMap, timeLaserInfoStamp, odometryFrame);

    // publish registered key frame
    if (pubRecentKeyFrame->get_subscription_count() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);

      *cloudOut += *transformPointCloud(laserCloudAllLastDS, &thisPose6D);

      publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish registered high-res raw cloud
    if (pubCloudRegisteredRaw->get_subscription_count() != 0)
    {
      pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
      PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);
      publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
    }
    // publish path
    if (pubOldPath->get_subscription_count() != 0)
    {
      globalOldPath.header.stamp = timeLaserInfoStamp;
      globalOldPath.header.frame_id = odometryFrame;
      pubOldPath->publish(globalOldPath);
    }
    if (pubNewPath->get_subscription_count() != 0)
    {
      globalNewPath.header.stamp = timeLaserInfoStamp;
      globalNewPath.header.frame_id = odometryFrame;
      pubNewPath->publish(globalNewPath);
    }    
    // 应该在这个地方发布闭环更新通知消息
    std_msgs::msg::Bool is_loop_msgs;
    is_loop_msgs.data = isLoopClosed;
    pubLoopClosed->publish(is_loop_msgs);
    //@todo publish suit submap
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
      RCLCPP_ERROR(get_logger(), "in 5s, save map failure, please check your map_dir config or call save map by manual!");
    }
    else
    {
      RCLCPP_INFO(get_logger(), "save map succeed!");
      // 向定位节点发布建图结束的命令 连续发布10帧
      for (int i = 0; i < 10; ++i)
      {
        std_msgs::msg::Empty msg;
        notisfy_local_pub->publish(msg);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }
    }
    // 关掉建图节点
    // closeMappingNode();
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::executors::MultiThreadedExecutor exec;

  auto MO = std::make_shared<mapOptimization>(options);
  exec.add_node(MO);

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");
  std::thread loopthread(&mapOptimization::loopClosureThread, MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);
  std::thread positionDetectionThread(&mapOptimization::positionDetThread, MO);
  exec.spin();

  rclcpp::shutdown();

  // loopthread.join();
  visualizeMapThread.join();

  return 0;
}
