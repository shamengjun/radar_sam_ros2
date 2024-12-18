#include "utility.h"
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

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class IMUPreintegration : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subWheelOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubCurrWheelOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::msg::Imu> imuQueOpt; // 优化后的imu值
    std::deque<sensor_msgs::msg::Imu> imuQueImu; // 原始imu值

    // 先前的状态
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_; // pose+ velocity
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;
    // 默认imu->lidar的外参旋转转换关系为0 0 0 1, 平移由配置文件的参数决定
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));


    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("radar_sam_imu_preintegration", options)
    {

        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);
        
        if (is_use_radar_odometry)
            subOdometry = create_subscription<nav_msgs::msg::Odometry>(
                "radar_sam/mapping/odometry_incremental",qos,
                std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        else
            subOdometry = create_subscription<nav_msgs::msg::Odometry>(
                wheelOdomTopic,qos,
                std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        // odomTopic -> odometry/imu
        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic + "_incremental", qos_imu);

        pubCurrWheelOdometry = create_publisher<nav_msgs::msg::Odometry>("debug_odom",100);

        pubImuPath = create_publisher<nav_msgs::msg::Path>("debug_path", 100);

        // 下面是预积分使用到的gtsam的一些参数配置 创建一个PreintegrationParams参数对象 包括加速计协方差 陀螺协方差
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);     // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);          // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);                                                               // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                             // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());   // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());                 // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        // 预积分前后的imu
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization
        
    }

    // 四元数转欧拉角
    Eigen::Vector3d QuaternionToEulerAngles(const Eigen::Quaterniond& q)
    {
        // 将四元数规范化
        Eigen::Vector3d angles;
        double norm = std::sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
        Eigen::Quaterniond normalized_q(q.w() / norm, q.x() / norm, q.y() / norm, q.z() / norm);

        // 欧拉角转换
        angles.x() = std::atan2(2 * (normalized_q.w() * normalized_q.x() + normalized_q.y() * normalized_q.z()),
                            1 - 2 * (normalized_q.x() * normalized_q.x() + normalized_q.y() * normalized_q.y()));
        angles.y() = std::asin(2 * (normalized_q.w() * normalized_q.y() - normalized_q.z() * normalized_q.x()));
        angles.z() = std::atan2(2 * (normalized_q.w() * normalized_q.z() + normalized_q.x() * normalized_q.y()),
                            1 - 2 * (normalized_q.y() * normalized_q.y() + normalized_q.z() * normalized_q.z()));
        return angles*180/3.1415926;
    }

    // 输入上一时刻四元数/这一时刻四元数/时间戳计算当前时刻角速度
    double yawVel(const Eigen::Quaterniond& q_Pre, const Eigen::Quaterniond& q_Now, const double& dt_q)
    {
        //角速度
        Eigen:: Vector3d euler_angles = QuaternionToEulerAngles(q_Pre);
        Eigen:: Vector3d euler_angles1 = QuaternionToEulerAngles(q_Now);
        Eigen::Vector3d angularVelocity = (euler_angles1 - euler_angles)/dt_q;
        return angularVelocity[2];
    }


    void resetOptimization()
    {
        std::cout << "reset optimization..." << std::endl;
        gtsam::ISAM2Params optParameters;

        //重线性化的阈值，当因子图中的变量与之前的估计值相差超过该阈值时
        //会重新线性化该点的因子图，并重新计算Hessian矩阵
        optParameters.relinearizeThreshold = 0.1;

        //relinearizeSkip设置了重线性化的间隔，即每经过多少个因子图优化步骤后进行一次重线性化
        optParameters.relinearizeSkip = 1;

        optimizer = gtsam::ISAM2(optParameters);

        //将因子图初始值设置为空图
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        //储存优化后的变量
        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        std::cout << "reset params" << std::endl;
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    // 这里是把lidar odo转到imu坐标系计算
    // 这里的odometry是相对值，即相对于上一个关键帧lidar/radar odometry的变化量
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        RCLCPP_DEBUG(get_logger(), "[imuPreintegration::IMUPreintegration]odometryHandler");

        static int frame_cnt = 0;
        if (!is_use_radar_odometry)
        {
            if ((++frame_cnt) % 10 != 0)
                return;
            nav_msgs::msg::Odometry curr_odom = (*odomMsg);
            pubCurrWheelOdometry->publish(curr_odom);
        }

        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);// odom的当前时间

        // make sure we have imu data to integrate 确保在当前时间下有imu数据
        if (imuQueOpt.empty())
            return;
        // RCLCPP_DEBUG_STREAM("radar_odom:\n" << *odomMsg);
        // 首先会从odom队列中取最新的pose作为先验，并做好时间对齐
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;

        if (!is_use_radar_odometry)
        {
           r_x = 0;
           r_y = 0;
           r_z = 0;
           r_w = 1.0;
        }
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false; // 判断是否退化

        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        // 0. initialize system
        if (systemInitialized == false)
        {
            std::cout << "Initialize system..." << std::endl;
            resetOptimization();

            // pop old IMU message
            // 去掉一些比较旧的imu数据, 只需要保证雷达odom时间戳在imu队列中间
            // 因为imu是高频数据, 这里是有效的
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp)  < currentCorrectionTime - delta_t) // 这里的delta_t = 0
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp) ;
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);//radar与imu的外参，计算imu在雷达坐标系下的状态
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise); //添加先验因子
            graphFactors.add(priorPose); //将其添加至因子图中
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0); // 初始速度置为0
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias(); // 初始Bias置为0
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once 这里好像并没有把优化结果取出来,但是graph结构更新了
            optimizer.update(graphFactors, graphValues);
            // 优化完后，将gragh清空
            graphFactors.resize(0);
            graphValues.clear();
            // 难道经过update()后，这里是prevBias_发生了改变?
            // 如果改变，那么就是说这个initial system的过程是为了估计初始的Bias
            // 如果没有改变，那这一段代码有什么作用呢?
            // 由下面一段代码可以看出，这里是为了估计出X(0)的状态，bias为0，但noise不为0，所以还是有意义的。
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            key = 1;
            systemInitialized = true;
            return;
        }

        // reset graph for speed
        // 这里的key作为计数器，在key超过100时重置整个因子图  减小计算压力 保存最后的噪声值
        if (key == 30)
        {
            // get updated noise before reset,重置前把最后噪声保存
            // std::cout << "resetOptimization..." << std::endl;
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // 1.integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            // 计算imu帧间时间差
            sensor_msgs::msg::Imu * thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            if (imuTime < currentCorrectionTime - delta_t) // 将radar odometry时间之前的imu数据均添加到预积分器中
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 100.0) : (imuTime - lastImuT_opt);
                // 进行预积分得到新的状态值,注意用到的是imu数据的加速度和角速度
                // 往imuIntegratorOpt_里面添加imu观测值，这个函数只是添加，并没有预测状态
                if (dt <= 0)
                {
                  RCLCPP_ERROR(get_logger(), "(1)dt <= 0. this maybe abnormal!");
                  dt = 1.0 / 100.0;
                }
                imuIntegratorOpt_->integrateMeasurement(
                    gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                    gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // add imu factor to graph //利用两帧之间的IMU数据完成了预积分后增加imu因子到因子图中
        //从imu预积分器中获取当前状态下的imu测量值
        const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorOpt_);   
        //这里认为当前状态和上一状态的bias是不变的，所以用上一状态的bias。
        //如果要对当前状态和上一状态的bias分别建模，就要用CombinedImuFactor。
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        // 还加入了pose factor,其实对应于作者论文中的因子图结构
        // 就是与imu因子一起的 Lidar odometry factor
        gtsam::Pose3 curPose  = lidarPose.compose(lidar2Imu); // radar odometry提供当前状态的先验值（还带了一手标定外参

        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values 根据先前的状态预测当前的状态，这里可以理解为直接积分，即由i时刻根据v a w计算出j时刻?

        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);//预测
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize 这个graph里包括一个imuFactor和PriorFactor
        optimizer.update(graphFactors, graphValues);
        optimizer.update();

        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 将优化后的结果取出来保存，用作下一次优化。
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_ = result.at<gtsam::Pose3>(X(key));
        prevVel_ = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_); // 保存当前的优化结果
        prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_); // 给积分器重新设置bia，所以每优化一次bias就要变一次。
        // check optimization
        // 如果优化后当前的状态太离谱，速度太大，意味是优化失败了，原因就是radar odometry这个先验因子没有把图约束住。
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }

        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom = prevBias_;
        // first pop imu message older than current correction data，将radar odometry时间前面的imu去掉，因为之前的数据已经用不了
        double lastImuQT = -1;

        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp)  < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp) ;
            
            imuQueImu.pop_front();
        }
        
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias 用优化后的bias的重置imuIntegratorImu_ 的bias注意有2个预积分器
            // 1个是imuIntegratorImu_这个通俗来讲是为了预测的，预测每个时刻的状态
            // 另一个是imuIntegratorOpt_这个是来优化的


            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);

            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                geometry_msgs::msg::PoseStamped pose11;
                gtsam::Pose3 imuPose11;
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);

            
                double imu_pre_xrv,imu_pre_yrv,imu_pre_zrv;

                double dt = (lastImuQT < 0) ? (1.0 / 100.0) : (imuTime - lastImuQT);
                if (dt < 0)
                    dt = (1.0 / 100.0);
                    
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

                lastImuQT = imuTime;
            }
            

        }
        RCLCPP_DEBUG(get_logger(), "[imuPreintegration::IMUPreintegration]odometryHandler END");

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);
        RCLCPP_DEBUG(get_logger(), "[imuPreintegration::IMUPreintegration]imuHandler");

        // 这里进行imu和lidar的坐标系变换  这里应该只变换旋转, 这里将lidar->imu之间的旋转外参考虑进来
        // 转换过来后的thisImu是imu在radar坐标系下的值
        sensor_msgs::msg::Imu thisImu = *imu_raw;

        // 获得时间间隔, 第一次为1/500,之后是两次imuTime间的差
        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 100.0) : (imuTime - lastImuT_imu); // lastImuT_imu初始值为-1
        // 解决gtsam中PreintegratedImuMeasurements::integrateMeasurement: dt <=0 的错误
        if (dt <= 0)
        {
            // RCLCPP_ERROR(get_logger(), "IMU timestamp jumped before %d, drop it!", dt);
            return;
        }
        // 2个imu队列中分别填充imu值
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 检查有没有执行过一次优化,这里需要先在odomhandler中优化一次后再进行该函数后续的工作
        // 也就是至少要接收到2帧的radar odometry才能完成一次优化
        if (doneFirstOpt == false)
            return;

        lastImuT_imu = imuTime;

        // integrate this single imu message
        // 进行预积分
        //std::cout << "dt:" << dt << std::endl;
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);

        // predict odometry
        // 根据预积分结果来预测odom  prevStateOdom和prevBiasOdom来自odometryHandler中
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar  这里是变换了平移
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position()); // 将预积分的结果拿出来
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();

        pubImuOdometry->publish(odometry); // 发布的是imu在lidar坐标下的pose


        RCLCPP_DEBUG(get_logger(), "[imuPreintegration::IMUPreintegration]imuHandlerend");

        static nav_msgs::msg::Path curr_imu_path;
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = thisImu.header.stamp;
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose = odometry.pose.pose;
        curr_imu_path.poses.push_back(pose_stamped);
        // 把lidarOdomTime1.0s以前的imuPath数据删除
        if (pubImuPath->get_subscription_count() != 0)
        {
            curr_imu_path.header.stamp = thisImu.header.stamp;
            curr_imu_path.header.frame_id = odometryFrame;
            pubImuPath->publish(curr_imu_path);
        }
    }
    
};

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto ImuP = std::make_shared<IMUPreintegration>(options);

    exec.add_node(ImuP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();

    return 0;
}
