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

const double RADIANS_PER_DEGREE = M_PI / 180.0;
const double DEGREES_PER_RADIAN = 180.0 / M_PI;
// WGS84 Parameters
#define WGS84_A 6378137.0        // major axis
#define WGS84_B 6356752.31424518  // minor axis
#define WGS84_F 0.0033528107     // ellipsoid flattening
#define WGS84_E 0.0818191908     // first eccentricity
#define WGS84_EP 0.0820944379    // second eccentricity

// UTM Parameters
#define UTM_K0 0.9996                   // scale factor
#define UTM_FE 500000.0                 // false easting
#define UTM_FN_N 0.0                    // false northing, northern hemisphere
#define UTM_FN_S 10000000.0             // false northing, southern hemisphere
#define UTM_E2 (WGS84_E * WGS84_E)      // e^2
#define UTM_E4 (UTM_E2 * UTM_E2)        // e^4
#define UTM_E6 (UTM_E4 * UTM_E2)        // e^6
#define UTM_EP2 (UTM_E2 / (1 - UTM_E2))  // e'^2

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

class vcuGpsDrivingMapping : public ParamServer
{
public:
    std::mutex mtx;
    std::mutex gpsMtx;
    std::mutex vcuMtx;
    std::mutex mtxGPS;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;         // sub vcu
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGpsOdometry;      // sub gps
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr finish_mapping_sub;     // sub save_map topic
    rclcpp::Subscription<gpal_msgs::msg::VcuData>::SharedPtr subVcuData;          // sub vcuData gear
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPSRaw;       // sub original gps
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointCloud; // sub radar_points
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPSInit;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOptOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubOptPath;
    rclcpp::Publisher<gpal_vision_msgs::msg::OverlayText>::SharedPtr pubOverlayText;
    rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pubOptGps;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pubUpdate;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRadarCloudMap;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph::shared_ptr gtSAMgraph;
    gtsam::Values graphValues;
    ISAM2 *isam;
    Values::shared_ptr isamCurrentEstimate; // isam的当前值
    Values::shared_ptr initialEstimate;

    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 用来存所有状态优化后的值
    std::vector<double> keyFramePoseTimestamp;
    bool systemInitialized = false;
    float transformTobeMapped[6]; // 存放优化前状态的估计值
    float initialGPSTransTobeMapped[6];
    float lasttransformTobeMapped[6];
    bool aLoopIsClosed = false;
    nav_msgs::msg::Path globalPath;
    nav_msgs::msg::Odometry lastGpsOdom;
    double currentTime;

    std::deque<nav_msgs::msg::Odometry> gpsOdom;

    Eigen::MatrixXd poseCovariance; // 位姿协方差

    GeographicLib::LocalCartesian geo_converter;

    string fileName;
    string fileNamePcd;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

    // 存放vcuData话题中的数据
    std::deque<gpal_msgs::msg::VcuData> vcuDataQueue;
    // 存放档位id的队列
    std::vector<int> vcu_gear;
    std::deque<sensor_msgs::msg::NavSatFix> gpsRawQueue;
    std::map<int, std::vector<double>> gps_key_map; 
    std::ofstream save_gps_tum_pose;
    std::ofstream save_trajectory;

    // Driving Mapping
    vector<pcl::PointCloud<PointType>::Ptr> allCloudKeyFrames;
    pcl::PointCloud<PointTypePose>::Ptr allKeyPoses6D;
    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    std::vector<double> mappingKeyPoseTimestamp; // 也可以直接在记录的关键帧位姿中进行处理,推算点云位姿
    map<int, PointTypePose> mapKeyPoses6D;
    pcl::VoxelGrid<PointType> downSizeFilterMap;
    std::mutex pointMtx;
    std::mutex vcuImuMtx;
    nav_msgs::msg::Odometry gpsInitOdom;
    int index = 0;
    vcuGpsDrivingMapping(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_vcu_gps_driving_mapping", options)
    {

        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "vcu_imu_odom", qos,
            std::bind(&vcuGpsDrivingMapping::odometryHandler, this, std::placeholders::_1));

        subGpsOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "gps_odom", qos, std::bind(&vcuGpsDrivingMapping::gpsOdometryHander, this, std::placeholders::_1));

        // 结束建图 保存轨迹+gp
        finish_mapping_sub = create_subscription<std_msgs::msg::Empty>(
            "finish_map", 1, std::bind(&vcuGpsDrivingMapping::finishMappingSub, this, std::placeholders::_1));

        subVcuData = create_subscription<gpal_msgs::msg::VcuData>(
            "vcu_data", 100, std::bind(&vcuGpsDrivingMapping::vcuDataHandler, this, std::placeholders::_1));

        subGPSRaw = create_subscription<sensor_msgs::msg::NavSatFix>(
            gps_topic_name, 10, std::bind(&vcuGpsDrivingMapping::gpsRawInfoHandler, this, std::placeholders::_1));

        subPointCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&vcuGpsDrivingMapping::pointCloudInfoHandler, this, std::placeholders::_1)); 

        subGPSInit = create_subscription<nav_msgs::msg::Odometry>(
            "init_odom", qos, std::bind(&vcuGpsDrivingMapping::gpsInitHander, this, std::placeholders::_1));

        // odomTopic -> odometry/imu
        pubOptOdometry = create_publisher<nav_msgs::msg::Odometry>("opt_odom", qos_imu);
        pubOdometry = create_publisher<nav_msgs::msg::Odometry>("curr_odom", qos_imu);

        pubOptPath = create_publisher<nav_msgs::msg::Path>("opt_path", qos_imu);

        pubOverlayText = create_publisher<gpal_vision_msgs::msg::OverlayText>("state_overlay", qos);

        pubOptGps = create_publisher<sensor_msgs::msg::NavSatFix>("opt_gps", qos);
        pubUpdate = create_publisher<std_msgs::msg::Bool>("updatePose", 1);
        pubRadarCloudMap = create_publisher<sensor_msgs::msg::PointCloud2>("map_global", 1);

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1; // isam2的参数设置 决定什么时候重新初始化
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        allKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        gtSAMgraph.reset(new gtsam::NonlinearFactorGraph());
        initialEstimate.reset(new gtsam::Values());
        isamCurrentEstimate.reset(new gtsam::Values());
        std::string currTimeStr = getCurrentTime();
        fileName = "traj-" + currTimeStr + ".traj";
        fileNamePcd = "pcd-" + currTimeStr + ".pcd";
        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    bool saveFrame()
    {
        if (cloudKeyPoses6D->points.empty())
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

    void addOdomFactor()
    {
        if (cloudKeyPoses6D->points.empty())
        {
            // 第一帧时，初始化gtsam参数
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 先验 0是key 第二个参数是先验位姿，最后一个参数是噪声模型，如果没看错的话，先验应该默认初始化成0了
            gtSAMgraph->add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 添加节点的初始估计值
            initialEstimate->insert(0, trans2gtsamPose(transformTobeMapped));
        }
        else
        {
            // 之后是添加二元的因子
            // noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // c pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); lasttransformTobeMapped
            gtsam::Pose3 poseFrom = trans2gtsamPose(lasttransformTobeMapped);
            gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);
            gtsam::Pose3 relPose = poseFrom.between(poseTo);
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtSAMgraph->add(BetweenFactor<Pose3>(cloudKeyPoses6D->size() - 1, cloudKeyPoses6D->size(), relPose, odometryNoise));
            initialEstimate->insert(cloudKeyPoses6D->size(), poseTo);
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
        if (cloudKeyPoses6D->points.empty())
            return;
        else
        {
            // if (poseDistance(cloudKeyPoses6D->front(), cloudKeyPoses6D->back()) < gpsDisctanceThreshold) //5m
            //     return;
        }

        // last gps position
        static PointType lastGPSPoint;

        // cout <<  fixed << setprecision(8) << "curr_time:" << currentTime << endl;
        while (!gpsOdom.empty())
        {
            // cout << fixed << setprecision(8) << stamp2Sec(gpsOdom.front().header.stamp) << endl;
            if (stamp2Sec(gpsOdom.front().header.stamp) < currentTime - 0.1) // 0.2
            {
                // message too old
                gpsOdom.pop_front();
            }
            else if (stamp2Sec(gpsOdom.front().header.stamp) > currentTime + 0.1) // 0.2
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
                //  意味着添加的gps的因子的噪声至少为1.0
                Vector3 << noise_x, noise_y, max(noise_x,noise_y);
                // Vector3 << 2.0 , 2.0, 2.0;
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses6D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph->add(gps_factor);
                aLoopIsClosed = true;                       
                break;
            }
        }
    }

    void correctPoses()
    {
        if (cloudKeyPoses6D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            allKeyPoses6D->clear();
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> correct poses.\033[0m");
            // clear path
            globalPath.poses.clear();
            // update key poses
            // 这里直接拿更新后因子图的值
            int numPoses = isamCurrentEstimate->size();
#pragma omp parallel for num_threads(numberOfCores) // 多线程并行
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses6D->points[i].x = isamCurrentEstimate->at<Pose3>(i).translation().x();
                cloudKeyPoses6D->points[i].y = isamCurrentEstimate->at<Pose3>(i).translation().y();
                cloudKeyPoses6D->points[i].z = isamCurrentEstimate->at<Pose3>(i).translation().z();
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate->at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate->at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate->at<Pose3>(i).rotation().yaw();

                Eigen::Vector3d lla;
                geo_converter.Reverse(cloudKeyPoses6D->points[i].x, cloudKeyPoses6D->points[i].y, cloudKeyPoses6D->points[i].z, lla[0], lla[1], lla[2]);
                cloudKeyPoses6D->points[i].latitude = lla[0];
                cloudKeyPoses6D->points[i].longitude = lla[1];
                cloudKeyPoses6D->points[i].altitude = lla[2];

                updatePath(cloudKeyPoses6D->points[i]);
                int id = cloudKeyPoses6D->points[i].intensity;
                if(mapKeyPoses6D.find(id) != mapKeyPoses6D.end())
                {
                    mapKeyPoses6D[id] =cloudKeyPoses6D->points[i];
                    allKeyPoses6D->push_back(cloudKeyPoses6D->points[i]);
                }
            }
            std_msgs::msg::Bool is_loop_msgs;
            is_loop_msgs.data = aLoopIsClosed;
            pubUpdate->publish(is_loop_msgs);
            aLoopIsClosed = false;

        }
    }

    void updatePath(const PointTypePose &pose_in)
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
        globalPath.poses.push_back(pose_stamped);
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

    // gps经纬度转UTM
    void LonLat2UTM(double longitude, double latitude, double& UTME, double& UTMN, std::string & UTMZone)
    {
        double Lat = latitude;
        double Long = longitude;

        double a = WGS84_A;
        double eccSquared = UTM_E2;
        double k0 = UTM_K0;

        double LongOrigin;
        double eccPrimeSquared;
        double N, T, C, A, M;
        double LongTemp =
            (Long + 180) - static_cast<int>((Long + 180) / 360) * 360 - 180;

        double LatRad = Lat * RADIANS_PER_DEGREE;
        double LongRad = LongTemp * RADIANS_PER_DEGREE;
        double LongOriginRad;
        int ZoneNumber;

        ZoneNumber = static_cast<int>((LongTemp + 180) / 6) + 1;
        if (Lat >= 56.0 && Lat < 64.0 && LongTemp >= 3.0 && LongTemp < 12.0) {
        ZoneNumber = 32;
        }
        // Special zones for Svalbard
        if (Lat >= 72.0 && Lat < 84.0) {
        if (LongTemp >= 0.0 && LongTemp < 9.0) {
            ZoneNumber = 31;
        } else if (LongTemp >= 9.0 && LongTemp < 21.0) {
            ZoneNumber = 33;
        } else if (LongTemp >= 21.0 && LongTemp < 33.0) {
            ZoneNumber = 35;
        } else if (LongTemp >= 33.0 && LongTemp < 42.0) {
            ZoneNumber = 37;
        }
        }

        // +3 puts origin in middle of zone
        LongOrigin = (ZoneNumber - 1) * 6 - 180 + 3;
        LongOriginRad = LongOrigin * RADIANS_PER_DEGREE;

        char LetterDesignator;

        if ((84 >= Lat) && (Lat >= 0)) {
        LetterDesignator = 'N';
        }else if ((0 > Lat) && (Lat >= -80)) {
        LetterDesignator = 'S';
        } else {
        // 'Z' is an error flag, the Latitude is outside the UTM limits
        LetterDesignator = 'Z';
        }

        char zone_buf[] = {0, 0, 0, 0};
        snprintf(
        zone_buf, sizeof(zone_buf), "%c%d",
        LetterDesignator, ZoneNumber & 0x3fU);
        UTMZone = std::string(zone_buf);

        eccPrimeSquared = (eccSquared) / (1 - eccSquared);

        N = a / sqrt(1 - eccSquared * sin(LatRad) * sin(LatRad));
        T = tan(LatRad) * tan(LatRad);
        C = eccPrimeSquared * cos(LatRad) * cos(LatRad);
        A = cos(LatRad) * (LongRad - LongOriginRad);

        M = a *
            ((1 - eccSquared / 4 - 3 * eccSquared * eccSquared / 64 -
            5 * eccSquared * eccSquared * eccSquared / 256) *
            LatRad -
            (3 * eccSquared / 8 + 3 * eccSquared * eccSquared / 32 +
            45 * eccSquared * eccSquared * eccSquared / 1024) *
            sin(2 * LatRad) +
            (15 * eccSquared * eccSquared / 256 +
            45 * eccSquared * eccSquared * eccSquared / 1024) *
            sin(4 * LatRad) -
            (35 * eccSquared * eccSquared * eccSquared / 3072) * sin(6 * LatRad));

        UTME = static_cast<double>(
            k0 * N *
            (A + (1 - T + C) * A * A * A / 6 +
            (5 - 18 * T + T * T + 72 * C - 58 * eccPrimeSquared) * A * A * A *
            A * A / 120) +
            500000.0);

        UTMN = static_cast<double>(
            k0 *
            (M + N * tan(LatRad) *
            (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A * A * A * A / 24 +
            (61 - 58 * T + T * T + 600 * C - 330 * eccPrimeSquared) * A *
            A * A * A * A * A / 720)));

        if (Lat < 0) {
            // 10000000 meter offset for southern hemisphere
            UTMN += 10000000.0;
        }
    }


    bool saveMap(std::string save_dir, float resolution)
    {
        string saveMapDirectory;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        saveMapDirectory = std::getenv("HOME") + save_dir + "/" + getCurrentTime() + "_" + park_name;
        cout << "Save destination: " << saveMapDirectory << endl;
        int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
        // save key frame transformations
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses6D);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
        for (int i = 0; i < (*cloudKeyPoses6D).size(); ++i)
        {
            save_trajectory.open(saveMapDirectory+"/trajectory.traj", std::ofstream::out | std::ofstream::app);
 
            // 将原点的utm坐标写入该文件
            double cartesian_x = 0;
            double cartesian_y = 0;
            std::string cartesian_zone_tmp;
            LonLat2UTM(gpsInitOdom.pose.pose.position.y, gpsInitOdom.pose.pose.position.x, cartesian_x, cartesian_y, cartesian_zone_tmp);
            save_trajectory << fixed << setprecision(8) << cartesian_zone_tmp << " " << cartesian_x << " " << cartesian_y << " " << std::endl;
            if(i == 0)
            {
                save_trajectory << fixed << setprecision(8) << cloudKeyPoses6D->points[i].time << " "
                        << cloudKeyPoses6D->points[i].x << " "
                        << cloudKeyPoses6D->points[i].y << " "
                        << cloudKeyPoses6D->points[i].z << " "
                        << 0 << " "
                        << 0 << " "
                        << 0 << " "
                        << cloudKeyPoses6D->points[i].latitude << " "
                        << cloudKeyPoses6D->points[i].longitude << " "
                        << cloudKeyPoses6D->points[i].altitude << endl;                            
            }
            else
            {
                double yaww = atan2(cloudKeyPoses6D->points[i].y - cloudKeyPoses6D->points[i-1].y,cloudKeyPoses6D->points[i].x - cloudKeyPoses6D->points[i-1].x);

                save_trajectory << fixed << setprecision(8) << cloudKeyPoses6D->points[i].time << " "
                        << cloudKeyPoses6D->points[i].x << " "
                        << cloudKeyPoses6D->points[i].y << " "
                        << cloudKeyPoses6D->points[i].z << " "
                        << cloudKeyPoses6D->points[i].roll << " "
                        << cloudKeyPoses6D->points[i].pitch << " "
                        << yaww << " "
                        << cloudKeyPoses6D->points[i].latitude << " "
                        << cloudKeyPoses6D->points[i].longitude << " "
                        << cloudKeyPoses6D->points[i].altitude << endl;                            
            }
            save_trajectory.close();

        }

        // 保存全局地图
        // create directory and remove old files;
        pcl::PointCloud<PointType>::Ptr globalAllCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalAllCloudDS(new pcl::PointCloud<PointType>());
        for (int i = 0; i < mapKeyPoses6D.size(); ++i) 
        {
            int key = std::next(std::begin(mapKeyPoses6D), i)->first;
            PointTypePose pose = std::next(std::begin(mapKeyPoses6D), i)->second;
            *globalAllCloud += *transformPointCloud(allCloudKeyFrames[i], &pose);
        }
        if (resolution != 0)
        {
            cout << "\n\nSave resolution: " << resolution << endl;
            // down-sample and save corner cloud
            downSizeFilterMap.setInputCloud(globalAllCloud);
            downSizeFilterMap.setLeafSize(resolution, resolution, resolution);
            downSizeFilterMap.filter(*globalAllCloudDS);
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

        // 压缩点云地图


        return true;
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(vcuImuMtx);
        static int frame_cnt = 0;
        if (!is_use_radar_odometry)
        {
            if ((++frame_cnt) % 3 != 0)
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
            // 将关键帧位姿存入
            if(!cloudKeyPoses6D->empty())
            {
                PointTypePose keyPose;
                keyPose = cloudKeyPoses6D->back();
                allKeyPoses6D->push_back(keyPose); 
                mappingKeyPoseTimestamp.push_back(keyPose.time);            
            }
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

        // 保存关键帧
        thisPose6D.x = latestEstimate.translation().x();
        thisPose6D.y = latestEstimate.translation().y();
        thisPose6D.z = latestEstimate.translation().z();
        thisPose6D.intensity = cloudKeyPoses6D->size(); // this can be used as index
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
        cloudKeyPoses6D->push_back(thisPose6D);
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
            gps_key_map.insert(std::pair<int, std::vector<double>>(cloudKeyPoses6D->size() - 1, gps_raw_vec));
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
        if (currentTime - lastCorrectTime > 30) // 10s correct一次
        {
            correctPoses();
            lastCorrectTime = currentTime;
        }

        if (pubOptPath->get_subscription_count() != 0) // 这里输出并不是10hz,
        {
            globalPath.header.stamp = odomMsg->header.stamp;
            globalPath.header.frame_id = odometryFrame;
            pubOptPath->publish(globalPath);
        }

        // 发布opt_odom
        if (pubOptOdometry->get_subscription_count() != 0)
        {
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
        }

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

        geometry_msgs::msg::TransformStamped ts;
        ts.header.frame_id = odometryFrame;
        ts.child_frame_id = "/gps";
        //ts.header.stamp = odomMsg->header.stamp;
        ts.header.stamp = this->get_clock()->now();
        // ts.transform.setOrigin(trans_base.getOrigin());
        // ts.transform.setRotation(trans_base.getRotation());
        ts.transform.translation.x = latestEstimate.translation().x();
        ts.transform.translation.y = latestEstimate.translation().y();
        ts.transform.translation.z = latestEstimate.translation().z();
        //ts.transform.rotation = tf2::toMsg(trans_base.getRotation());
        ts.transform.rotation.x = latestEstimate.rotation().toQuaternion().x();
        ts.transform.rotation.y = latestEstimate.rotation().toQuaternion().y();
        ts.transform.rotation.z = latestEstimate.rotation().toQuaternion().z();
        ts.transform.rotation.w = latestEstimate.rotation().toQuaternion().w();

        tfBroadcaster->sendTransform(ts);
    }

    void pointCloudInfoHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msgIn)
    {
        std::lock_guard<std::mutex> lock(pointMtx);
        cloudQueue.push_back(*msgIn);
    }

    void gpsOdometryHander(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(gpsMtx);
        gpsOdom.push_back(*odomMsg);

        if (cloudKeyPoses6D->points.empty())
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "gps inititialize.");

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

            // 获取初始的lla
            Eigen::Vector3d lla(odomMsg->pose.covariance[1], odomMsg->pose.covariance[2], odomMsg->pose.covariance[3]);
            geo_converter.Reset(lla[0], lla[1], lla[2]);

            systemInitialized = true;
        }
        return;
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

    void drivingMappingThread()
    {
        // 5s执行一次
        rclcpp::Rate rate(0.3);
        while (rclcpp::ok())
        {
            // 时间戳对齐
            rate.sleep();
            runDrivingMapping();
            publishGlobalMap();
            
        }
    }

    void runDrivingMapping()
    {
        std::lock_guard<std::mutex> pointLock(pointMtx);
        std::lock_guard<std::mutex> imuLock(vcuImuMtx);

        // 在关键帧位姿中寻找与点云帧匹配的位姿
        if(cloudKeyPoses6D->empty() || cloudQueue.empty())
        {
            return;
        }

        while(!cloudQueue.empty())
        {
            // 上一次在关键帧位姿队列中寻找的id：用于缩小搜寻范围
            double point_time;
            sensor_msgs::msg::PointCloud2 currentCloudMsg;
            pcl::PointCloud<PointType>::Ptr radarCloudIn(new pcl::PointCloud<PointType>());
            currentCloudMsg = cloudQueue.front();     // 拿到当前点云 
            cloudQueue.pop_front();
            pcl::moveFromROSMsg(currentCloudMsg, *radarCloudIn); // 转成ros消息 laserCloudIn
            point_time = stamp2Sec(currentCloudMsg.header.stamp);    
        
            for(int i = index; i < cloudKeyPoses6D->points.size(); ++i)
            {
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "i: %d, keyFramePoseTimestamp[i]: %f, point_time:%f", i, keyFramePoseTimestamp[i], point_time);
                RCLCPP_INFO(rclcpp::get_logger("rclcpp"), " keyFramePoseTimestamp[max]: %f", i, keyFramePoseTimestamp[cloudKeyPoses6D->points.size()-1]);
                std::cout << "index: " << index << std::endl;
                // 这里的0.05与0.03要按照车速相应的进行修正
                if(point_time < keyFramePoseTimestamp[i]-0.1)
                {
                    // continue;
                }
                else 
                {
                    if(fabs(point_time - keyFramePoseTimestamp[i]) < 0.03)
                    {
                        allCloudKeyFrames.push_back(radarCloudIn);
                        PointTypePose keyPose = cloudKeyPoses6D->points[i];
                        mapKeyPoses6D[keyPose.intensity] = keyPose;     
                        index = i/1.05;
                        continue;                        
                    }

                }
            }

        }

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
#pragma omp parallel for num_threads(numberOfCores) 
        for (int i = 0; i < mapKeyPoses6D.size(); ++i) 
        {
            int key = std::next(std::begin(mapKeyPoses6D), i)->first;
            PointTypePose pose = std::next(std::begin(mapKeyPoses6D), i)->second;
            *globalMapKeyFrames += *transformPointCloud(allCloudKeyFrames[i], &pose);
        }

        sensor_msgs::msg::PointCloud2 cloudMsgGlobalMapKeyFrames;
        pcl::toROSMsg(*globalMapKeyFrames, cloudMsgGlobalMapKeyFrames);
        rclcpp::Clock clock;
        cloudMsgGlobalMapKeyFrames.header.stamp = clock.now();
        cloudMsgGlobalMapKeyFrames.header.frame_id = "odom";
        pubRadarCloudMap->publish(cloudMsgGlobalMapKeyFrames);

    }

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto MO = std::make_shared<vcuGpsDrivingMapping>(options);
    exec.add_node(MO);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> VCU gps optmization Started.\033[0m");
    std::thread drivingMappingThread(&vcuGpsDrivingMapping::drivingMappingThread, MO);
    exec.spin();

    rclcpp::shutdown();
    drivingMappingThread.join();
    return 0;
}
