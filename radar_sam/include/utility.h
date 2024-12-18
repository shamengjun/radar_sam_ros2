#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#define PCL_NO_PRECOMPILE // !! BEFORE ANY PCL INCLUDE!! for custom define pointcloud format registration

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include "std_msgs/msg/int32.hpp"
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include "gpal_vision_msgs/msg/overlay_text.hpp"

#include <std_msgs/msg/empty.h>
#include <std_msgs/msg/int32.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include "pcl/registration/ndt.h"
#include "pcl/registration/gicp.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/compression/octree_pointcloud_compression.h> //for pointcloud compression
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
// #include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_eigen/tf2_eigen.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <vector>
#include <list>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <signal.h>

using namespace std;

typedef std::numeric_limits<double> dbl;

typedef pcl::PointXYZI PointType;
// 语义点云注册
// 车位点云注册
struct XYZRGBSemanticsInfo
{
    PCL_ADD_POINT4D;
    PCL_ADD_RGB;
    std::uint32_t label;
    float score;
    std::uint32_t t;
    std::uint32_t id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(XYZRGBSemanticsInfo,
                                  (float, x, x)(float, y, y)(float, z, z)(uint8_t, r, r)(uint8_t, g, g)(uint8_t, b, b)(uint32_t, label, label)(float, score, score)(uint32_t, t, t)
                                  (uint32_t, id, id))

typedef pcl::PointCloud<XYZRGBSemanticsInfo> SenmanticsInfoPointType;

enum class SensorType
{
    VELODYNE,
    OUSTER
};

class ParamServer : public rclcpp::Node
{
public:
    // Topics
    string pointCloudTopic;
    string imuTopic;
    string imuTopicOr;
    string odomTopic;
    string gpsTopic;
    string wheelOdomTopic;

    // Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;
    float gpsDisctanceThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int Radar_target_number;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    vector<std::string> mapping_nodes;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // voxel filter paprams
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;
    float mappingLoopLeafSize;
    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool loopClosureEnableFlag;
    float loopClosureFrequency;
    int surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    // add by zhaoyz
    int multi_frame;
    int loopIntervalFrames;
    float ndtEpsion, ndtStepSize, ndtResolution;
    int ndtMaxInter;
    string tumPoseFile;
    float filter_min_z;

    int submapOutputMethod;
    float submapOutputSearchRadius;
    int submapOutputContinueNumFrames;
    bool cropBoundingBox;
    float bbx_a;
    float bbx_b;
    float bbx_c;
    bool downSampleSubmap;
    float radiusSearchLeafSize;
    float continueNumberFrameLeafSize;
    int ndt_search_method;
    string time_elapsed_csv_file;
    bool add_wheel_odometry;

    bool is_convert_to_map;
    string gps_topic_name;
    bool is_use_radar_odometry;

    string map_save_dir;
    bool is_auto_save_map;
    bool map_compression;
    string park_name;
    bool is_save_tum_pose;
    bool is_save_align_info;
    string lane_topic_name;

    bool is_save_gps_pose;
    bool is_use_gps_z;
    int local_map_size;
    bool is_add_snr;
    bool is_use_gps_a;
    bool is_use_gps_c;

    bool is_save_z_value;

    // 2023.6.25 for graph
    bool is_global_optimization;
    bool is_visualization_graph;
    // 2023.7.25 for ending mapping
    int auto_ending_mapping_method;
    int start_end_dis;
    int start_end_time;
    int drive_rounds;
    int loop_nums;
    string m_fusion_topic_name;
    bool is_filter_cloud_map_pcd;

    double map_pcd_setRadiusSearch;
    int map_pcd_setMinNeighborsInRadius;
    int map_pcd_setMeanK;
    double map_pcd_setStddevMulThresh;

    // 24.1.3 for 记忆行车
    int mapping_mode;


    ParamServer(std::string node_name, const rclcpp::NodeOptions &options) : Node(node_name, options)
    {
        declare_parameter("m_fusion_topic_name", "/fusion_data");
        get_parameter("m_fusion_topic_name", m_fusion_topic_name);

        declare_parameter("pointCloudTopic", "/radar_points2");
        get_parameter("pointCloudTopic", pointCloudTopic);

        declare_parameter("imuTopic", "/imu");
        get_parameter("imuTopic", imuTopic);

        declare_parameter("imuTopicOr", "/imu_raw");
        get_parameter("imuTopicOr", imuTopicOr); 

        declare_parameter("odomTopic", "odometry/imu");
        get_parameter("odomTopic", odomTopic);

        declare_parameter("wheelOdomTopic", "/rs/odom_a");
        get_parameter("wheelOdomTopic", wheelOdomTopic);

        declare_parameter("gpsTopic", "odometry/gps");
        get_parameter("gpsTopic", gpsTopic);

        declare_parameter("lidarFrame", "base_link");
        get_parameter("lidarFrame", lidarFrame);

        declare_parameter("baselinkFrame", "base_link");
        get_parameter("baselinkFrame", baselinkFrame);

        declare_parameter("odometryFrame", "odom");
        get_parameter("odometryFrame", odometryFrame);

        declare_parameter("mapFrame", "map");
        get_parameter("mapFrame", mapFrame);

        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);

        declare_parameter("useGpsElevation", false);
        get_parameter("useGpsElevation", useGpsElevation);

        declare_parameter("gpsCovThreshold", 2.0);
        get_parameter("gpsCovThreshold", gpsCovThreshold);

        declare_parameter("poseCovThreshold", 25.0);
        get_parameter("poseCovThreshold", poseCovThreshold);

        declare_parameter("gpsDisctanceThreshold", 25.0);
        get_parameter("gpsDisctanceThreshold", gpsDisctanceThreshold);

        declare_parameter("savePCDDirectory", "/home/gpal/radar_sam/src/resultSave");
        get_parameter("savePCDDirectory", savePCDDirectory);

        declare_parameter("Radar_target_number", 2000);
        get_parameter("Radar_target_number", Radar_target_number);

        declare_parameter("imuAccNoise", 0.01);
        get_parameter("imuAccNoise", imuAccNoise);

        declare_parameter("imuGyrNoise", 0.001);
        get_parameter("imuGyrNoise", imuGyrNoise);

        declare_parameter("imuAccBiasN", 0.0002);
        get_parameter("imuAccBiasN", imuAccBiasN);

        declare_parameter("imuGyrBiasN", 0.00003);
        get_parameter("imuGyrBiasN", imuGyrBiasN);

        declare_parameter("imuGravity", 9.7964);
        get_parameter("imuGravity", imuGravity);

        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);

        double ida[] = {1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0};
        std::vector<double> id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);

        double zea[] = {0.0, 0.0, 0.0};
        std::vector<double> ze(zea, std::end(zea));
        declare_parameter("extrinsicTrans", ze);
        get_parameter("extrinsicTrans", extTransV);

        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);

        declare_parameter("mappingSurfLeafSize", 0.2);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("mappingLoopLeafSize", 0.5);
        get_parameter("mappingLoopLeafSize", mappingLoopLeafSize);

        declare_parameter("z_tollerance", FLT_MAX);
        get_parameter("z_tollerance", z_tollerance);

        declare_parameter("rotation_tollerance", FLT_MAX);
        get_parameter("rotation_tollerance", rotation_tollerance);

        declare_parameter("numberOfCores", 2);
        get_parameter("numberOfCores", numberOfCores);

        declare_parameter("mappingProcessInterval", 0.15);
        get_parameter("mappingProcessInterval", mappingProcessInterval);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);

        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);

        declare_parameter("surroundingKeyframeDensity", 1.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);

        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", false);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);

        declare_parameter("loopClosureFrequency", 1.0);
        get_parameter("loopClosureFrequency", loopClosureFrequency);

        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);

        declare_parameter("historyKeyframeSearchRadius", 10.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);

        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);

        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        declare_parameter("globalMapVisualizationSearchRadius", 1e3);
        get_parameter("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius);

        declare_parameter("globalMapVisualizationPoseDensity", 10.0);
        get_parameter("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity);

        declare_parameter("globalMapVisualizationLeafSize", 1.0);
        get_parameter("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize);

        declare_parameter("multi_frame", 4);
        get_parameter("multi_frame", multi_frame);

        declare_parameter("loopIntervalFrames", 10);
        get_parameter("loopIntervalFrames", loopIntervalFrames);

        declare_parameter("ndtEpsion", 0.01);
        get_parameter("ndtEpsion", ndtEpsion);

        declare_parameter("ndtStepSize", 0.1);
        get_parameter("ndtStepSize", ndtStepSize);

        declare_parameter("ndtResolution", 1.0);
        get_parameter("ndtResolution", ndtResolution);

        declare_parameter("ndtMaxInter", 100);
        get_parameter("ndtMaxInter", ndtMaxInter);

        declare_parameter("tumPoseFile", "radar_sam_pose_tum.txt");
        get_parameter("tumPoseFile", tumPoseFile);

        declare_parameter("filter_min_z", 0.1);
        get_parameter("filter_min_z", filter_min_z);

        declare_parameter("submapOutputMethod", 0);
        get_parameter("submapOutputMethod", submapOutputMethod);

        declare_parameter("submapOutputSearchRadius", 5.0);
        get_parameter("submapOutputSearchRadius", submapOutputSearchRadius);

        declare_parameter("submapOutputContinueNumberFrames", 10);
        get_parameter("submapOutputContinueNumberFrames", submapOutputContinueNumFrames);

        declare_parameter("cropBoundingBox", false);
        get_parameter("cropBoundingBox", cropBoundingBox);

        declare_parameter("bbx_a", 0.1);
        get_parameter("bbx_a", bbx_a);

        declare_parameter("bbx_b", 0.1);
        get_parameter("bbx_b", bbx_b);

        declare_parameter("bbx_c", 0.1);
        get_parameter("bbx_c", bbx_c);

        declare_parameter("downSampleSubmap", false);
        get_parameter("downSampleSubmap", downSampleSubmap);

        declare_parameter("radiusSearchLeafSize", 0.5);
        get_parameter("radiusSearchLeafSize", radiusSearchLeafSize);

        declare_parameter("continueNumberFrameLeafSize", 0.5);
        get_parameter("continueNumberFrameLeafSize", continueNumberFrameLeafSize);

        declare_parameter("ndt_search_method", 1);
        get_parameter("ndt_search_method", ndt_search_method);

        declare_parameter("time_elapsed_csv_file", std::string("radar_slam_time_elapsed.csv"));
        get_parameter("time_elapsed_csv_file", time_elapsed_csv_file);

        declare_parameter("add_wheel_odometry", false);
        get_parameter("add_wheel_odometry", add_wheel_odometry);

        declare_parameter("is_convert_to_map", true);
        get_parameter("is_convert_to_map", is_convert_to_map);

        declare_parameter("gps_topic_name", std::string("rs/gps"));
        get_parameter("gps_topic_name", gps_topic_name);

        declare_parameter("is_use_radar_odometry", true);
        get_parameter("is_use_radar_odometry", is_use_radar_odometry);

        declare_parameter("savePCD", false);
        get_parameter("savePCD", savePCD);

        declare_parameter("map_save_dir", "/home/zhaoyz/map_dir");
        get_parameter("map_save_dir", map_save_dir);

        declare_parameter("is_auto_save_map", false);
        get_parameter("is_auto_save_map", is_auto_save_map);

        declare_parameter("map_compression", false);
        get_parameter("map_compression", map_compression);

        declare_parameter("park_name", "youdu_factory");
        get_parameter("park_name", park_name);

        declare_parameter("auto_ending_mapping_method", 0);
        get_parameter("auto_ending_mapping_method", auto_ending_mapping_method);

        declare_parameter("start_end_dis", 3);
        get_parameter("start_end_dis", start_end_dis);

        declare_parameter("start_end_time", 1);
        get_parameter("start_end_time", start_end_time);

        declare_parameter("drive_rounds", 100);
        get_parameter("drive_rounds", drive_rounds);

        declare_parameter("loop_nums", 30);
        get_parameter("loop_nums", loop_nums);

        // declare_parameter("radar_sam_mapping_nodes", "/radar_sam_imageProjection /radar_sam_transFusionData /radar_sam_imuPreintegration /radar_sam_rviz /radar_sam_transformFusion /robot_state_publisher /radar_sam_mapOptmization");

        string zxx[] = {"/radar_sam_imageProjection", "/radar_sam_transFusionData",
                        "/radar_sam_imuPreintegration", "/radar_sam_rviz",
                        "/radar_sam_transformFusion", "/robot_state_publisher", "/radar_sam_mapOptmization"};
        std::vector<string> zex(zxx, std::end(zxx));
        declare_parameter("radar_sam_mapping_nodes", zex);
        if (!get_parameter("radar_sam_mapping_nodes", mapping_nodes))
        {
            RCLCPP_ERROR(get_logger(), "failed to get radar_sam_mapping_nodes from param_server");
        }

        declare_parameter("is_save_tum_pose", false);
        get_parameter("is_save_tum_pose", is_save_tum_pose);

        declare_parameter("is_save_align_info", false);
        get_parameter("is_save_align_info", is_save_align_info);

        declare_parameter("is_save_gps_pose", false);
        get_parameter("is_save_gps_pose", is_save_gps_pose);

        declare_parameter("is_use_gps_z", false);
        get_parameter("is_use_gps_z", is_use_gps_z);

        declare_parameter("is_add_snr", false);
        get_parameter("is_add_snr", is_add_snr);

        declare_parameter("is_use_gps_a", false);
        get_parameter("is_use_gps_a", is_use_gps_a);

        declare_parameter("is_use_gps_c", false);
        get_parameter("is_use_gps_c", is_use_gps_c);

        declare_parameter("is_save_z_value", false);
        get_parameter("is_save_z_value", is_save_z_value);

        declare_parameter("is_global_optimization", false);
        get_parameter("is_global_optimization", is_global_optimization);

        declare_parameter("is_visualization_graph", false);
        get_parameter("is_visualization_graph", is_visualization_graph);

        declare_parameter("local_map_size", 15);
        get_parameter("local_map_size", local_map_size);

        declare_parameter("is_filter_cloud_map_pcd", true);
        get_parameter("is_filter_cloud_map_pcd", is_filter_cloud_map_pcd);

        declare_parameter("map_pcd_setRadiusSearch", 0.5);
        get_parameter("map_pcd_setRadiusSearch", map_pcd_setRadiusSearch);
        declare_parameter("map_pcd_setMinNeighborsInRadius", 5);
        get_parameter("map_pcd_setMinNeighborsInRadius", map_pcd_setMinNeighborsInRadius);
        declare_parameter("map_pcd_setMeanK", 5);
        get_parameter("map_pcd_setMeanK", map_pcd_setMeanK);
        declare_parameter("map_pcd_setStddevMulThresh", 1.0);
        get_parameter("map_pcd_setStddevMulThresh", map_pcd_setStddevMulThresh);

        declare_parameter("mapping_mode", 1);
        get_parameter("mapping_mode", mapping_mode);


        usleep(100);
    }

    int getPidImpl(std::string node_name)
    {
        FILE *fstream = NULL;
        char buff[1024];
        memset(buff, 0, sizeof(buff));
        std::string cmd = "ps aux | grep " + node_name + " | grep -v grep | awk '{print $2}'";
        RCLCPP_WARN(get_logger(), "cmd:%s", cmd.c_str());
        if (NULL == (fstream = popen(cmd.c_str(), "r")))
        {
            fprintf(stderr, "execute command failed: %s", strerror(errno));
            return -1;
        }
        while (NULL != fgets(buff, sizeof(buff), fstream))
        {
            RCLCPP_WARN(get_logger(), "%s", buff);
        }
        pclose(fstream);
        std::string str = buff;
        int pid = atoi(str.c_str());
        RCLCPP_WARN(get_logger(), "%s %s pid: %d", str.c_str(), node_name.c_str(), pid);
        return pid;
    }

    static bool killByPid(unsigned int pid)
    {
        kill(pid, SIGTERM);
        return true;
    }

    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu &imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y,
                            imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y,
                            imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
        {
            RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }

        return imu_out;
    }
};

template <typename T>
double stamp2Sec(const T &stamp)
{
    return rclcpp::Time(stamp).seconds();
}

template <typename T>
double stamp2Sec(const T &stamp1, const T &stamp2)
{
    rclcpp::Duration duration = stamp1 - stamp2;
    return duration.seconds();
}

sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

// 重载发布车位语义的点云
sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template <typename T>
double ROS_TIME(T msg)
{
    return stamp2Sec(msg->header.stamp);
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::msg::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    if (thisImuMsg == nullptr) 
    {
        // 进行错误处理，比如抛出异常或返回错误码
        return;
    }

    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

float pointDistance(PointType p)
{
    double d = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return static_cast<float>(d);
}

float pointDistance(PointType p1, PointType p2)
{
    double d = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
    return static_cast<float>(d);
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

string getCurrentTime()
{
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    char buf[100] = {0};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&now));
    return buf;
}

rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth),
    qos_profile);

rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false};

auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth),
    qos_profile_imu);

rmw_qos_profile_t qos_profile_gps{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false};

auto qos_gps = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_gps.history,
        qos_profile_gps.depth),
    qos_profile_gps);

rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth),
    qos_profile_lidar);

std::string padZeros(int val, int num_digits = 6)
{
    std::ostringstream out;
    out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
    return out.str();
}

void publishOverlayText(const std::string &str,
                        rclcpp::Publisher<gpal_vision_msgs::msg::OverlayText>::SharedPtr pub_ov,
                        const std::string fg_color = "b")
{
    gpal_vision_msgs::msg::OverlayText ov_msg;
    ov_msg.text = str;
    ov_msg.fg_color.a = 1.0;
    // https://github.com/jkk-research/colors
    if (fg_color == "r") // red
    {
        ov_msg.fg_color.r = 0.96f;
        ov_msg.fg_color.g = 0.22f;
        ov_msg.fg_color.b = 0.06f;
    }
    else if (fg_color == "g") // green
    {
        ov_msg.fg_color.r = 0.30f;
        ov_msg.fg_color.g = 0.69f;
        ov_msg.fg_color.b = 0.31f;
    }
    else if (fg_color == "b") // blue
    {
        ov_msg.fg_color.r = 0.02f;
        ov_msg.fg_color.g = 0.50f;
        ov_msg.fg_color.b = 0.70f;
    }
    else if (fg_color == "k") // black
    {
        ov_msg.fg_color.r = 0.19f;
        ov_msg.fg_color.g = 0.19f;
        ov_msg.fg_color.b = 0.23f;
    }
    ov_msg.width = 500;
    ov_msg.height = 60;
    ov_msg.font = 20;
    pub_ov->publish(ov_msg);
}
#endif
