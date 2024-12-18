#include "utility.h"
#include "radar_sam/msg/cloud_info.hpp"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/statistical_outlier_removal.h"

#define DEBUG_RADAR_30_m 0 // 23.7.8 debug for radar 30m model

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    std::mutex imuLock;
    std::mutex odoLock;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;

    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<radar_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;

    std::deque<sensor_msgs::msg::Imu> imuQueue;

    std::deque<nav_msgs::msg::Odometry> odomQueue;

    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    sensor_msgs::msg::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;                 // imu队列的当前位置指针
    bool firstPointFlag;               // firstPointFlag 还不知道什么意思
    Eigen::Affine3f transStartInverse; // 一个变换，不知道什么意思

    pcl::PointCloud<PointType>::Ptr laserCloudIn; // 点云输入

    pcl::PointCloud<PointType>::Ptr fullCloud;      // 全点云 不知道什么意思
    pcl::PointCloud<PointType>::Ptr extractedCloud; // 提取的点云 不知道什么意思

    int deskewFlag; // 去畸变的flag

    radar_sam::msg::CloudInfo cloudInfo;
    double timeScanCur; // 对于激光 一帧点云有开始和结束两个时间 对于毫米波 我们只使用timeScanCur
    double timeScanEnd; // 这个之后一定不再使用 这个timeScanEnd和timeScanCur一样的
    std_msgs::msg::Header cloudHeader;

public:
    ImageProjection(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_imageProjection", options), deskewFlag(0)
    {

        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        // imu的订阅 应该不变
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        // subOdom = nh.subscribe<nav_msgs::msg::Odometry>("rs/odom_a", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);

        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/deskew/cloud_deskewed", 10);
        pubLaserCloudInfo = create_publisher<radar_sam::msg::CloudInfo>("radar_sam/deskew/cloud_info", qos);

        // pubImage = nh.advertise<sensor_msgs::msg::Image> ("radar_sam/deskew/imaggg", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        fullCloud->points.resize(Radar_target_number);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();

        imuPointerCur = 0;
        firstPointFlag = true;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection() {}

    // 只是把imu的数据添加到queue中
    void    imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        // std::cout << "============imuHandler============" << std::endl;
        sensor_msgs::msg::Imu thisImu = *imuMsg;
        // std::cout << "imu time :" << stamp2Sec(thisImu.header.stamp) << std::endl;
        // std::cout << "imu time :" <<thisImu.header.stamp.sec <<"===" << thisImu.header.stamp.nanosec << std::endl;
        // std::cout << setprecision(12) << thisImu.header.stamp.toSec() << std::endl;
        // 做个判断如果当前的imu时间戳跳变的话，则丢掉该帧
        if (!imuQueue.empty() && stamp2Sec(imuQueue.back().header.stamp) > stamp2Sec(thisImu.header.stamp) + 0.1)
        {
            RCLCPP_INFO(get_logger(),"imu Queue front time=%.8f timeScanCur=%.8f",stamp2Sec(imuQueue.back().header.stamp), stamp2Sec(thisImu.header.stamp));
            RCLCPP_ERROR(get_logger(), "IMU stamp jump before %d", stamp2Sec(imuQueue.back().header.stamp) - stamp2Sec(thisImu.header.stamp));
            return;
        }
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    // 原始的radar点云的处理入口函数
    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg) // TODO:
    {
        // 去噪处理 测试下来 速度太慢了
        //  pcl::PointCloud<PointType>::Ptr after_dbscan = runDbscan(laserCloudMsg);
        //  1.获取点云中最大值和最小值，获取一定范围内点云；
        //  2.将车一圈的点云滤除去；
        //  3.将动态的点剔除
        //  std::cout << "============cloudHandler============" << std::endl;
        if (!cachePointCloud(laserCloudMsg)) // 缓存点云, 主要是当前点云的时间戳的赋值
            return;

        if (!deskewInfo()) // 找到radar当前时间对应的imu和odom信息，估计出当前帧的pose
            return;

        projectPointCloud(); // 投影点云，转化为rangeMat的形式,这个rangeMat目前并没有用到

        cloudExtraction(); // 只是将full cloud的值赋给了 extracted cloud

        publishClouds();

        resetParameters();
    }

    // radar数据预处理，主要是时间戳的获取
    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr &laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg); // 放进点云队列
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        // std::cout << "laserCloudMsg time" << laserCloudMsg->header.stamp.sec
        //           << " nsec:" << laserCloudMsg->header.stamp.nanosec << std::endl;

        currentCloudMsg = std::move(cloudQueue.front());     // 拿到当前点云
        cloudQueue.pop_front();                              // 队列里pop一个
        pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn); // 转成ros消息 laserCloudIn
        // std::cout << "raw point cloud size:" << laserCloudIn->points.size() << std::endl;
        // 在这里做去噪处理比较方便一些
        // 获取最大最小距离点
        PointType min, max;
        pcl::getMinMax3D(*laserCloudIn, min, max); // 这里最大值104, 最小-42  max: 104.214 44.5402 57.355
                                                   //                        min: -47.8037 -60.9644 0
#if DEBUG_RADAR_30_m
        std::cout << "points size:" << laserCloudIn->points.size() << std::endl;
        std::cout << "max:[" << max.x << "," << max.y << "," << max.z << "] min:["
                  << min.x << "," << min.y << "," << min.z << "]" << std::endl;
#endif
        // 统计点云z轴上的值
        //  double max_z = 0, min_z = 0, total_z = 0.0;
        //  for (int i = 0; i < laserCloudIn->points.size(); ++i)
        //  {
        //     if (laserCloudIn->points[i].z > max_z)
        //       max_z = laserCloudIn->points[i].z;
        //     if (laserCloudIn->points[i].z < min_z)
        //       min_z = laserCloudIn->points[i].z;
        //      total_z += laserCloudIn->points[i].z;
        //      std::cout << laserCloudIn->points[i].z << " ";
        //  }
        //  std::cout << std::endl;
        //  std::cout << "max_z:" << max_z << " min_z:" << min_z << " average_z:" << total_z / laserCloudIn->points.size() << std::endl;
#if 1
        pcl::RadiusOutlierRemoval<PointType> outstrem;
        outstrem.setInputCloud(laserCloudIn);
        outstrem.setRadiusSearch(0.5); // 1m范围内至少有2个点
        outstrem.setMinNeighborsInRadius(3);
        // outstrem.setKeepOrganized(true);
        outstrem.filter(*laserCloudIn);
        // std::cout << "after radius removal:" << laserCloudIn->points.size() << std::endl;
        // 将x轴前2m, 后1m, y轴 左右1m, z轴1m内的点滤除
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud(laserCloudIn);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-1.0, 2.0);
        pass.setFilterLimitsNegative(true);
        pass.filter(*laserCloudIn);
        // //std::cout << "after pass through x:" << laserCloudIn->points.size() << std::endl;
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-1.0, 1.0);
        pass.setFilterLimitsNegative(true);
        pass.filter(*laserCloudIn);
        // std::cout << "after pass through y:" << laserCloudIn->points.size() << std::endl;

        pass.setFilterFieldName("z");
        pass.setFilterLimits(filter_min_z, 50.0); // 10.0
        pass.setFilterLimitsNegative(false);
        pass.filter(*laserCloudIn);
        // std::cout << "after pass through:" << laserCloudIn->points.size() << std::endl;
        // 移除离群点滤波
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(laserCloudIn);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1);
        sor.filter(*laserCloudIn);
#endif
        // std::cout << "after statical outliers remove:" << laserCloudIn->points.size() << std::endl;
        //  get timestamp 获取时间
        cloudHeader = currentCloudMsg.header;
        timeScanCur = stamp2Sec(cloudHeader.stamp); // TODO:权宜之计
        // timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
        // timeScanEnd = timeScanCur + 0.2; //这里我的想法是，让timeScanEnd等于下一帧的时间，现在获取这个可能会出错，所以暂时加入magic number
        timeScanEnd = timeScanCur;
        // 让timeScanEnd = timeScanCur+（两帧时间差）
        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        if (imuQueue.empty())
        {
            RCLCPP_ERROR(get_logger(), "IMU Queue is empty!");
            return false;
        }

        if (/*imuQueue.empty() ||*/ stamp2Sec(imuQueue.front().header.stamp) > timeScanCur || stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd) // TODO:
        {
            RCLCPP_DEBUG(get_logger(), "Waiting for IMU data ...");
            // RCLCPP_DEBUG_STREAM("imu Queue front time = " << (imuQueue.front().header.stamp.toSec()));
            // RCLCPP_DEBUG_STREAM("timeScanCur = " << timeScanCur);
            RCLCPP_DEBUG(get_logger(), "imu Queue front time=%.8f timeScanCur=%.8f", stamp2Sec(imuQueue.front().header.stamp), timeScanCur);
            // RCLCPP_DEBUG_STREAM("imu Queue back time = " << (imuQueue.back().header.stamp.toSec()));
            // RCLCPP_DEBUG_STREAM("timeScanEnd = " << timeScanEnd);
            RCLCPP_DEBUG(get_logger(), "imu Queue back time=%.8f timeScanEnd=%.8f", stamp2Sec(imuQueue.back().header.stamp), timeScanEnd);
            return false;
        }

        imuDeskewInfo(); // 看下面的代码，似乎这里不用改，但好像和timeScanCur 有关系 待定TODO:

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;

        // 将距离当前radar时间0.1s以前的imu旧的数据丢弃掉
        while (!imuQueue.empty())
        {
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.001)
                imuQueue.pop_front(); // 如果imu队列里的front时间比timeScanCur提前太多，就pop掉，直到接近
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i) // 遍历imu队列
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur) // 在imu队列里，现在应该只有第一项在timeScanCur前了吧？所以应该只做一次
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            // 反正是获得了当前帧的rpy估计，存在cloudInfo里

            if (currentImuTime > timeScanEnd + 0.01) // imu数据太新
                break;

            // 在imu队列第一帧，初始化imuRot为0
            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                cloudInfo.imu_roll_init = 0;
                cloudInfo.imu_pitch_init = 0;
                cloudInfo.imu_yaw_init = 0;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity在这之后的帧，首先得到角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation然后实际上做了个积分 这样算能准吗？ 这里保存imuRotX是为了后面去畸变使用的
            double timeDiff = currentImuTime - imuTime[imuPointerCur - 1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }
        // 最终，imuRot成为了在这帧期间imu返回的角度变化，注意，每帧初始都设置成0，所以这是个相对值
        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imu_available = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;

        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::msg::Odometry startOdomMsg;

        // 遍历odo队列，找到比timeScanCur大的startOdomMsg
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        } // 最终是找到odomMsg时间正好比timeScancur大的地方

        // 转成tf
        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

        // 获得此时rpy
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // 这就是我们在mapOptimization里的初始猜测，所以我们知道，这个猜测是来源于odom
        // Initial guess used in mapOptimization
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // 如果odom的最后一项时间都比ScanEnd小，那就没必要找了
        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        nav_msgs::msg::Odometry endOdomMsg;

        // 遍历 找到比timeScanEnd大的endOdomMsg
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }

        // 如果位姿的不确定性不一样，就return，不理解这是用来做什么的
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // transBegin
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // transEnd
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 得到相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;
    }

    void projectPointCloud()
    {

        int cloudSize = laserCloudIn->points.size();
        int index = 0;
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // changed by zhaoyz 23.6.1 add radar snr information
            if (is_add_snr)
                thisPoint.intensity = laserCloudIn->points[i].intensity;

            fullCloud->points[index] = thisPoint;
            index++;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        int cloudSize = laserCloudIn->points.size();
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < cloudSize; ++i)
        {
            // save extracted cloud
            extractedCloud->push_back(fullCloud->points[i]);
            // size of extracted cloud
            ++count;
        }
    }

    // 发布cloudinfo
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
