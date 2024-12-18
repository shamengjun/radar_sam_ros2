#include "utility.h"
#include "gpal_msgs/msg/vcu_data.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <std_msgs/msg/string.hpp>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <Eigen/Dense> //四元数转旋转向量
#include <std_msgs/msg/float32.hpp>
#include <GeographicLib/LocalCartesian.hpp> //包含头文件
#include "utm.h"

#define IS_CONVERT_UTM false

struct PointXYZIRPYTLLA
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    double latitude;
    double longitude;
    double altitude;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYTLLA,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time)(double, latitude, latitude)(double, longitude, longitude)(double, altitude, altitude))

typedef PointXYZIRPYTLLA PointTypePose;
using namespace gtsam;

class vcuGPSOptimization : public ParamServer
{
public:
    std::mutex mtx;
    std::mutex gpsMtx;
    std::mutex vcuMtx;
    std::mutex mtxGPS;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;    // sub vcu
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGpsOdometry; // sub gps
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr finish_mapping_sub;
    rclcpp::Subscription<gpal_msgs::msg::VcuData>::SharedPtr subVcuData;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPSRaw;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPSInit;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOptOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubOptPath;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pubCorrectMsg;
    rclcpp::Publisher<gpal_vision_msgs::msg::OverlayText>::SharedPtr pubOverlayText;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pubOptGps;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pubUpdate;
    nav_msgs::msg::Odometry lastGpsOdom;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph::shared_ptr gtSAMgraph;
    ISAM2 *isam;
    gtsam::Values graphValues;
    Values::shared_ptr isamCurrentEstimate; // isam的当前值
    Values::shared_ptr initialEstimate;

    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses3D; // 用来存所有状态优化后的值
    std::vector<double> keyFramePoseTimestamp;
    bool systemInitialized = false;

    float transformTobeMapped[6]; // 存放优化前状态的估计值
    float initialGPSTransTobeMapped[6];
    float lasttransformTobeMapped[6];

    bool aLoopIsClosed = false;

    nav_msgs::msg::Path globalPath;

    double currentTime;

    std::deque<nav_msgs::msg::Odometry> gpsOdom;
    nav_msgs::msg::Odometry gpsInitOdom;
    Eigen::MatrixXd poseCovariance; // 位姿协方差

    GeographicLib::LocalCartesian geo_converter;

    string fileName;
    string fileTumName;
    string fileNamePcd;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    Pose3 currEstimate;

    // 存放vcuData话题中的数据
    std::deque<gpal_msgs::msg::VcuData> vcuDataQueue;
    // 存放档位id的队列
    std::vector<int> vcu_gear;
    std::deque<sensor_msgs::msg::NavSatFix> gpsRawQueue;
    std::map<int, std::vector<double>> gps_key_map;
    std::ofstream save_gps_tum_pose;
    std::ofstream save_trajectory;
    vcuGPSOptimization(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_vcu_gps_optimization", options)
    {

        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "vcu_imu_odom", qos,
            std::bind(&vcuGPSOptimization::odometryHandler, this, std::placeholders::_1));

        subGpsOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "gps_odom", qos, std::bind(&vcuGPSOptimization::gpsOdometryHander, this, std::placeholders::_1));

        subGPSInit = create_subscription<nav_msgs::msg::Odometry>(
            "init_odom", qos, std::bind(&vcuGPSOptimization::gpsInitHander, this, std::placeholders::_1));
        // 结束建图 保存轨迹+gp
        finish_mapping_sub = create_subscription<std_msgs::msg::Empty>(
            "finish_map", 1, std::bind(&vcuGPSOptimization::finishMappingSub, this, std::placeholders::_1));

        subVcuData = create_subscription<gpal_msgs::msg::VcuData>(
            "vcu_data", 100, std::bind(&vcuGPSOptimization::vcuDataHandler, this, std::placeholders::_1));

        subGPSRaw = create_subscription<sensor_msgs::msg::NavSatFix>(
            gps_topic_name, 10, std::bind(&vcuGPSOptimization::gpsRawInfoHandler, this, std::placeholders::_1));

        // odomTopic -> odometry/imu
        pubOptOdometry = create_publisher<nav_msgs::msg::Odometry>("opt_odom", qos_imu);

        pubOdometry = create_publisher<nav_msgs::msg::Odometry>("curr_odom", qos_imu);

        pubOptPath = create_publisher<nav_msgs::msg::Path>("opt_path", qos_imu);

        pubCorrectMsg = create_publisher<std_msgs::msg::Empty>("correct_pose", qos);

        pubOverlayText = create_publisher<gpal_vision_msgs::msg::OverlayText>("state_overlay", qos);

        pubOptGps = create_publisher<sensor_msgs::msg::NavSatFix>("opt_gps", qos);
        pubUpdate = create_publisher<std_msgs::msg::Bool>("updatePose", 1);

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1; // isam2的参数设置 决定什么时候重新初始化
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointTypePose>());

        gtSAMgraph.reset(new gtsam::NonlinearFactorGraph());
        initialEstimate.reset(new gtsam::Values());
        isamCurrentEstimate.reset(new gtsam::Values());
        std::string currTimeStr = getCurrentTime();
        fileName = "traj-" + currTimeStr + ".traj";
        fileNamePcd = "pcd-" + currTimeStr + ".pcd";
        fileTumName = "tum-" + currTimeStr + ".txt";
        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            std::cout << "[saveFrame] first frame--->return" << std::endl;
            return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses3D->back());
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

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧时，初始化gtsam参数
            // noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished()); // rad*rad, meter*meter
            // 先验 0是key 第二个参数是先验位姿，最后一个参数是噪声模型，如果没看错的话，先验应该默认初始化成0了
            // gtSAMgraph->add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            gtSAMgraph->add(PriorFactor<Pose3>(0, trans2gtsamPose(initialGPSTransTobeMapped), priorNoise));
            // 添加节点的初始估计值
            // initialEstimate->insert(0, trans2gtsamPose(transformTobeMapped));
            initialEstimate->insert(0, trans2gtsamPose(initialGPSTransTobeMapped));
        }
        else
        {
            // 之后是添加二元的因子
            // noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // c pclPointTogtsamPose3(cloudKeyPoses3D->points.back()); lasttransformTobeMapped
            gtsam::Pose3 poseFrom = trans2gtsamPose(lasttransformTobeMapped);
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            gtsam::Pose3 relPose = poseFrom.between(poseTo);
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtSAMgraph->add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), relPose, odometryNoise));
            initialEstimate->insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    double poseDistance(PointTypePose p1, PointTypePose p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
    }

    void addGPSFactor()
    {
        if (gpsOdom.empty())
        {
            return;
        }

        // wait for system initialized and settles down
        // 系统初始化且位移一段时间了再考虑要不要加gpsfactor
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // if (poseDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < gpsDisctanceThreshold) //5m
            //     return;
        }

        // last gps position
        static PointType lastGPSPoint;

        // cout <<  fixed << setprecision(8) << "curr_time:" << currentTime << endl;
        while (!gpsOdom.empty())
        {
            // cout << fixed << setprecision(8) << stamp2Sec(gpsOdom.front().header.stamp) << endl;
            if (stamp2Sec(gpsOdom.front().header.stamp) < currentTime - 0.5) // 0.2
            {
                // message too old
                gpsOdom.pop_front();
            }
            else if (stamp2Sec(gpsOdom.front().header.stamp) > currentTime + 0.2) // 0.2
            {
                // message too new
                break;
            }
            else // 找到timeLaserInfoCur前后0.2s内的gps数据, 这里的0.2应该根据实际gps的频率来定。
            {
                // cout << "find gps data." << endl;
                nav_msgs::msg::Odometry thisGPS = gpsOdom.front();
                gpsOdom.pop_front();

                // GPS too noisy, skip 23.3.3目对于rs的这里是固定的0.1
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // 目前这里的noise_x,noise_y,noise_z都是0,因为gps的原始消息里也是0
                // 所以需要给gps的协方差字段赋值，这里相当于百分百的相信gps消息了
                // gpsCovThreshold目前设置的值为2.0
                // std::cout << "gps noise:" << noise_x << " " << noise_y << " covThreshold:" << gpsCovThreshold;
                // if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold) // 0.135335 0.2
                //     continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                // if (pointDistance(curGPSPoint, lastGPSPoint) < gpsDisctanceThreshold) // 至少间隔5m以上 1.0
                //     continue;
                // else
                //     lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);

                // std::cout << " add one gps factor." << gps_x << " " << gps_y << " " << gps_z << std::endl;
                //   意味着添加的gps的因子的噪声至少为1.0
                //  Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                Vector3 << 1e-3, 1e-3, 1e-3;
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph->add(gps_factor);
                aLoopIsClosed = true;

                break;
            }
        }
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> correct poses.\033[0m");
            //  clear path
            globalPath.poses.clear();
            // update key poses
            // 这里直接拿更新后因子图的值
            int numPoses = isamCurrentEstimate->size();
#pragma omp parallel for num_threads(numberOfCores) // 多线程并行
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate->at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate->at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate->at<Pose3>(i).translation().z();
                cloudKeyPoses3D->points[i].roll = isamCurrentEstimate->at<Pose3>(i).rotation().roll();
                cloudKeyPoses3D->points[i].pitch = isamCurrentEstimate->at<Pose3>(i).rotation().pitch();
                cloudKeyPoses3D->points[i].yaw = isamCurrentEstimate->at<Pose3>(i).rotation().yaw();

                Eigen::Vector3d lla;

                geo_converter.Reverse(cloudKeyPoses3D->points[i].x, cloudKeyPoses3D->points[i].y, cloudKeyPoses3D->points[i].z, lla[0], lla[1], lla[2]);
                cloudKeyPoses3D->points[i].latitude = lla[0];
                cloudKeyPoses3D->points[i].longitude = lla[1];
                cloudKeyPoses3D->points[i].altitude = lla[2];

                updatePath(cloudKeyPoses3D->points[i]);
            }
            // 发布更新消息
            std_msgs::msg::Empty msg;
            pubCorrectMsg->publish(msg);
            std_msgs::msg::Bool is_loop_msgs;
            is_loop_msgs.data = aLoopIsClosed;
            pubUpdate->publish(is_loop_msgs);
#if 0
            // 定时保存轨迹，每隔2min刷新一次
            static double lastSaveTime = currentTime;
            if (currentTime - lastSaveTime > 2 * 60)
            {
                // 1. 保存成.traj
                ofstream traj_file(fileName);
                if (traj_file)
                {
                    // 将原点的utm坐标写入该文件
                
                    // LonLat2UTM(gpsInitOdom.pose.pose.position.y, gpsInitOdom.pose.pose.position.x, cartesian_x, cartesian_y, cartesian_zone_tmp);
                    long zone = 0;
                    char hes = 0;
                    double e;
                    double n;
                    Convert_Geodetic_To_UTM(gpsInitOdom.pose.pose.position.x * M_PI / 180, gpsInitOdom.pose.pose.position.y * M_PI / 180, 
                                           &zone, &hes, &e, &n);

                    std::string cartesian_hes;
                    cartesian_hes = hes;
                    std::string cartesian_zone = std::to_string(zone);
                    traj_file << fixed << setprecision(8) << cartesian_hes + cartesian_zone << " " << e << " " << n << " " << std::endl;
                    // std::cout <<gpsInitOdom.pose.pose.position.y << " " << gpsInitOdom.pose.pose.position.x << std::endl;
                    for (int i = 0; i < cloudKeyPoses3D->points.size(); ++i)
                    {
                        if (i == 0)
                        {
                            traj_file << fixed << setprecision(8) << cloudKeyPoses3D->points[i].time << " "
                                      << cloudKeyPoses3D->points[i].x << " "
                                      << cloudKeyPoses3D->points[i].y << " "
                                      << cloudKeyPoses3D->points[i].z << " "
                                      << 0 << " "
                                      << 0 << " "
                                      << 0 << " "
                                      << cloudKeyPoses3D->points[i].latitude << " "
                                      << cloudKeyPoses3D->points[i].longitude << " "
                                      << cloudKeyPoses3D->points[i].altitude << endl;
                        }
                        else
                        {
                            double yaww = atan2(cloudKeyPoses3D->points[i].y - cloudKeyPoses3D->points[i - 1].y, cloudKeyPoses3D->points[i].x - cloudKeyPoses3D->points[i - 1].x);

                            traj_file << fixed << setprecision(8) << cloudKeyPoses3D->points[i].time << " "
                                      << cloudKeyPoses3D->points[i].x << " "
                                      << cloudKeyPoses3D->points[i].y << " "
                                      << cloudKeyPoses3D->points[i].z << " "
                                      << cloudKeyPoses3D->points[i].roll << " "
                                      << cloudKeyPoses3D->points[i].pitch << " "
                                      << yaww << " "
                                      << cloudKeyPoses3D->points[i].latitude << " "
                                      << cloudKeyPoses3D->points[i].longitude << " "
                                      << cloudKeyPoses3D->points[i].altitude << endl;
                        }
                    }
                    traj_file.close();
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "\033[1;31m----> file open failure.\033[0m");
                }

                // 2. 保存成pcd格式
                pcl::io::savePCDFileBinary(fileNamePcd, *cloudKeyPoses3D);

                lastSaveTime = currentTime;

                // 3. 保存成tum
                ofstream tum_file(fileTumName);
                if (tum_file)
                {
                    for (int i = 0; i < cloudKeyPoses3D->points.size(); ++i)
                    {
                        tum_file << fixed << setprecision(8) << cloudKeyPoses3D->points[i].time << " "
                                 << cloudKeyPoses3D->points[i].x << " "
                                 << cloudKeyPoses3D->points[i].y << " "
                                 << cloudKeyPoses3D->points[i].z << " "
                                 << cloudKeyPoses3D->points[i].roll << " "
                                 << cloudKeyPoses3D->points[i].pitch << " "
                                 << cloudKeyPoses3D->points[i].yaw << " "
                                 << endl;
                    }
                    tum_file.close();
                }
                else
                {
                    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "\033[1;31m----> file open failure.\033[0m");
                }
            }
#endif
            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose &pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
        pose_stamped.header.frame_id = odometryFrame;
        // 这里转换成utm填充进去
        if (IS_CONVERT_UTM)
        {
            long zone = 0;
            char hes = 0;
            double e;
            double n;
            Convert_Geodetic_To_UTM(pose_in.latitude * M_PI / 180, pose_in.longitude * M_PI / 180, &zone, &hes, &e, &n);
            // cout << fixed << setprecision(8) << e << " " << n << endl;
            pose_stamped.pose.position.x = e;
            pose_stamped.pose.position.y = n;
            pose_stamped.pose.position.z = pose_in.z;
        }
        else
        {
            pose_stamped.pose.position.x = pose_in.x;
            pose_stamped.pose.position.y = pose_in.y;
            pose_stamped.pose.position.z = pose_in.z;
        }

        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        static int frame_cnt = 0;
        if (!is_use_radar_odometry)
        {
            if ((++frame_cnt) % 5 != 0)
                return;
        }

        currentTime = stamp2Sec(odomMsg->header.stamp); // odom的当前时间

        // 更新 transformtoMapped 首次由gps的位置来对齐
        tf2::Quaternion orientation;
        tf2::fromMsg(odomMsg->pose.pose.orientation, orientation);

        // 获得此时rpy
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        transformTobeMapped[0] = roll;
        transformTobeMapped[1] = pitch;
        transformTobeMapped[2] = yaw;
        transformTobeMapped[3] = odomMsg->pose.pose.position.x;
        transformTobeMapped[4] = odomMsg->pose.pose.position.y;
        transformTobeMapped[5] = odomMsg->pose.pose.position.z;

        // 等待用gps的位置进行初始化
        if (!systemInitialized)
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "waiting for gps data initializing.");
            return;
        }

        // 添加odometry between factor
        if (saveFrame())
        {
            addOdomFactor();
            // gps factor
            addGPSFactor();
        }
        else // 保证10hz的输出， 在上一个关键帧的基础上，将vcu的相对信息添加上去即可
        {
            gtsam::Pose3 poseFrom = trans2gtsamPose(lasttransformTobeMapped); // 只有添加关键帧了才更新
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);       // 实时在更新
            gtsam::Pose3 relPose = poseFrom.between(poseTo);                  // 相对上一关键帧的相对变换

            // 上一关键帧的pose 保存在 latestEstimate
            gtsam::Pose3 latestEstimate = isamCurrentEstimate->at<Pose3>(isamCurrentEstimate->size() - 1);

            gtsam::Pose3 currEstimate = latestEstimate * relPose;

            // 转化为odometry 发布出来
            if (pubOdometry->get_subscription_count() != 0)
            {
                nav_msgs::msg::Odometry odom;
                odom.header.stamp = odomMsg->header.stamp;
                odom.header.frame_id = odometryFrame;
                odom.child_frame_id = baselinkFrame;
                odom.pose.pose.position.x = currEstimate.translation().x();
                odom.pose.pose.position.y = currEstimate.translation().y();
                odom.pose.pose.position.z = currEstimate.translation().z();
                odom.pose.pose.orientation.x = currEstimate.rotation().toQuaternion().x();
                odom.pose.pose.orientation.y = currEstimate.rotation().toQuaternion().y();
                odom.pose.pose.orientation.z = currEstimate.rotation().toQuaternion().z();
                odom.pose.pose.orientation.w = currEstimate.rotation().toQuaternion().w();
                pubOdometry->publish(odom);
            }
            // 转化成gps的消息发出来，投影到google地图上，看是否能对得齐，实际上也看不出来车道级别的对齐
            // 只能看个大概

            return;
        }

        // 优化
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

            if (aLoopIsClosed == true) // 是否加入了gps的factor
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

        // 获取优化结果
        latestEstimate = isamCurrentEstimate->at<Pose3>(isamCurrentEstimate->size() - 1);
        currEstimate = latestEstimate;

        // 保存关键帧
        thisPose6D.x = latestEstimate.translation().x();
        thisPose6D.y = latestEstimate.translation().y();
        thisPose6D.z = latestEstimate.translation().z();
        thisPose6D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw = latestEstimate.rotation().yaw();
        thisPose6D.time = currentTime;
        // 填充经纬高
        Eigen::Vector3d lla;
        geo_converter.Reverse(thisPose6D.x, thisPose6D.y, thisPose6D.z, lla[0], lla[1], lla[2]);
        thisPose6D.latitude = lla[0];
        thisPose6D.longitude = lla[1];
        thisPose6D.altitude = lla[2];
        cloudKeyPoses3D->push_back(thisPose6D);

        keyFramePoseTimestamp.push_back(currentTime);
        // int gear = associateKeyframeWithVcuDAata(currentTime);
        int gear = 0;
        vcu_gear.push_back(gear);
        while (!gpsRawQueue.empty())
        {
            if (stamp2Sec(gpsRawQueue.front().header.stamp) < currentTime - 1.01)
                gpsRawQueue.pop_front();
            else
                break;
        }
        if (!gpsRawQueue.empty())
        {
            sensor_msgs::msg::NavSatFix gps_msg = gpsRawQueue.front();
            std::vector<double> gps_raw_vec;
            gps_raw_vec.push_back(gps_msg.latitude);
            gps_raw_vec.push_back(gps_msg.longitude);
            gps_raw_vec.push_back(gps_msg.altitude);
            gps_key_map.insert(std::pair<int, std::vector<double>>(cloudKeyPoses3D->size() - 1, gps_raw_vec));
        }

        for (int i = 0; i < 6; ++i)
            lasttransformTobeMapped[i] = transformTobeMapped[i];

        // 更新transformTobeMapped
        // transformTobeMapped[0] = latestEstimate.rotation().roll();
        // transformTobeMapped[1] = latestEstimate.rotation().pitch();
        // transformTobeMapped[2] = latestEstimate.rotation().yaw();
        // transformTobeMapped[3] = latestEstimate.translation().x();
        // transformTobeMapped[4] = latestEstimate.translation().y();
        // transformTobeMapped[5] = latestEstimate.translation().z();
        // 更新整个path
        updatePath(thisPose6D);

        // 如果添加了gps的factor,则更新整个轨迹
        // 定时correct
        static double lastCorrectTime = currentTime;
        if (currentTime - lastCorrectTime > 1) // 10s correct一次
        {
            correctPoses();
            lastCorrectTime = currentTime;
        }

        // if (pubOptPath->get_subscription_count() != 0) // 这里输出并不是10hz,
        // {
        globalPath.header.stamp = odomMsg->header.stamp;
        globalPath.header.frame_id = odometryFrame;
        pubOptPath->publish(globalPath);
        //}

        // 发布opt_odom
        // if (pubOptOdometry->get_subscription_count() != 0)
        // {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = odomMsg->header.stamp;
        odom.header.frame_id = odometryFrame;
        odom.child_frame_id = baselinkFrame;
        odom.pose.pose.position.x = latestEstimate.translation().x();
        odom.pose.pose.position.y = latestEstimate.translation().y();
        odom.pose.pose.position.z = latestEstimate.translation().z();
        odom.pose.pose.orientation.x = latestEstimate.rotation().toQuaternion().x();
        odom.pose.pose.orientation.y = latestEstimate.rotation().toQuaternion().y();
        odom.pose.pose.orientation.z = latestEstimate.rotation().toQuaternion().z();
        odom.pose.pose.orientation.w = latestEstimate.rotation().toQuaternion().w();
        pubOptOdometry->publish(odom);
        //}

        // 发布opt_gps
        if (pubOptGps->get_subscription_count() != 0)
        {
            sensor_msgs::msg::NavSatFix gps;
            gps.header.stamp = odomMsg->header.stamp;
            gps.header.frame_id = "gps";
            gps.latitude = lla[0];
            gps.longitude = lla[1];
            gps.altitude = lla[2];
            pubOptGps->publish(gps);
        }
    }

    void gpsOdometryHander(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(gpsMtx);
        gpsOdom.push_back(*odomMsg);

        if (cloudKeyPoses3D->points.empty())
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "gps inititialize.");

            tf2::Quaternion orientation;
            tf2::fromMsg(odomMsg->pose.pose.orientation, orientation);

            // 获得此时rpy
            double roll, pitch, yaw;
            tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

            // transformTobeMapped[0] = roll;
            // transformTobeMapped[1] = pitch;
            // transformTobeMapped[2] = yaw;
            // transformTobeMapped[3] = odomMsg->pose.pose.position.x;
            // transformTobeMapped[4] = odomMsg->pose.pose.position.y;
            // transformTobeMapped[5] = odomMsg->pose.pose.position.z;

            initialGPSTransTobeMapped[0] = roll;
            initialGPSTransTobeMapped[1] = pitch;
            initialGPSTransTobeMapped[2] = yaw;
            initialGPSTransTobeMapped[3] = odomMsg->pose.pose.position.x;
            initialGPSTransTobeMapped[4] = odomMsg->pose.pose.position.y;
            initialGPSTransTobeMapped[5] = odomMsg->pose.pose.position.z;

            // 获取初始的lla
            Eigen::Vector3d lla(odomMsg->pose.covariance[1], odomMsg->pose.covariance[2], odomMsg->pose.covariance[3]);
            geo_converter.Reset(lla[0], lla[1], lla[2]);

            std::cout << fixed << setprecision(8) << "lla:" << lla[0] << " " << lla[1] << " " << lla[2] << std::endl;
            std::cout << "local:" << transformTobeMapped[3] << " " << transformTobeMapped[4] << " " << transformTobeMapped[5] << std::endl;
            std::cout << "rotation:" << initialGPSTransTobeMapped[0] << " " << initialGPSTransTobeMapped[1] << " " << initialGPSTransTobeMapped[2] << std::endl;

            systemInitialized = true;
        }
        return;
    }

    void publishTfThread()
    {
        rclcpp::Rate rate(10);
        while (rclcpp::ok())
        {
            rate.sleep();
            geometry_msgs::msg::TransformStamped ts;
            ts.header.frame_id = odometryFrame;
            ts.child_frame_id = "/gps";
            // ts.header.stamp = odomMsg->header.stamp;
            ts.header.stamp = this->get_clock()->now();
            // ts.transform.setOrigin(trans_base.getOrigin());
            // ts.transform.setRotation(trans_base.getRotation());
            ts.transform.translation.x = currEstimate.translation().x();
            ts.transform.translation.y = currEstimate.translation().y();
            ts.transform.translation.z = currEstimate.translation().z();
            // ts.transform.rotation = tf2::toMsg(trans_base.getRotation());
            ts.transform.rotation.x = currEstimate.rotation().toQuaternion().x();
            ts.transform.rotation.y = currEstimate.rotation().toQuaternion().y();
            ts.transform.rotation.z = currEstimate.rotation().toQuaternion().z();
            ts.transform.rotation.w = currEstimate.rotation().toQuaternion().w();

            tfBroadcaster->sendTransform(ts);
        }
    }

    void vcuDataHandler(const gpal_msgs::msg::VcuData::SharedPtr vcuDataMsg)
    {
        std::lock_guard<std::mutex> lock(vcuMtx);
        vcuDataQueue.push_back(*vcuDataMsg);
    }

    void gpsRawInfoHandler(const sensor_msgs::msg::NavSatFix::SharedPtr gps_msg)
    {
        mtxGPS.lock();
        gpsRawQueue.push_back(*gps_msg);
        mtxGPS.unlock();
    }

    void gpsInitHander(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        gpsInitOdom = *odomMsg;
    }

    int associateKeyframeWithVcuDAata(double time)
    {
        std::lock_guard<std::mutex> lock(vcuMtx);
        if (vcuDataQueue.empty())
        {
            RCLCPP_WARN(this->get_logger(), "vcuDataQueue is empty!");
            return 0;
        }
        // 进行时间同步 这里的VcuData频率较高
        while (!vcuDataQueue.empty())
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
        }
    }

    bool saveMap(std::string save_dir, float resolution)
    {
        string saveMapDirectory;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        saveMapDirectory = std::getenv("HOME") + save_dir;
        cout << "Save destination: " << saveMapDirectory << endl;

        // save key frame transformations
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses3D);
        for (int i = 0; i < (*cloudKeyPoses3D).size(); ++i)
        {
            save_trajectory.open(saveMapDirectory + "/trajectory.traj", std::ofstream::out | std::ofstream::app);
            save_trajectory.precision(18);
            double roll = (*cloudKeyPoses3D).points[i].roll;
            double pitch = (*cloudKeyPoses3D).points[i].pitch;
            double yaw = (*cloudKeyPoses3D).points[i].yaw;
            tf2::Quaternion quattf;
            quattf.setRPY(roll, pitch, yaw);

            geometry_msgs::msg::Quaternion quat;
            quat.x = quattf.x();
            quat.y = quattf.y();
            quat.z = quattf.z();
            quat.w = quattf.w();
            save_trajectory << keyFramePoseTimestamp[i] << "," << (*cloudKeyPoses3D).points[i].x << "," << (*cloudKeyPoses3D).points[i].y << "," << (*cloudKeyPoses3D).points[i].z << ","
                            << quat.x << "," << quat.y << "," << quat.z << "," << quat.w << "," << vcu_gear[i]
                            << std::endl;
            save_trajectory.close();
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

        return true;
    }
};

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto vcuGpsOpt = std::make_shared<vcuGPSOptimization>(options);

    exec.add_node(vcuGpsOpt);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> VCU gps optmization Started.\033[0m");
    std::thread publishTfThread(&vcuGPSOptimization::publishTfThread, vcuGpsOpt);
    exec.spin();

    rclcpp::shutdown();

    return 0;
}
