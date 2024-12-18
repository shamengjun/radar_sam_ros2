
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include "gpal_msgs/msg/vcu_data.hpp"
#include "gpal_msgs/msg/vcu_odom.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <vector>
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

#include "utility.h"
#include <Eigen/Dense>

std::string vcu_data = "/vcu_data";
std::string vcu_odom = "/vcu_odom";
std::string vcu_nav_odom = "/vcu_nav_odom";
std::string gps_data = "/gps";
std::string imu_data = "/imu_raw";

double L_FRONT = 1.605; // 两轮胎的横向距离
double L_REAR = 1.605;
double R_FRONT = 2.7; // 两轮胎的纵向距离
std::ofstream ofsVcuPath;
bool is_imu_raw = false;

// Eigen::Matrix4d imu_to_base;
// imu_to_base << 1.0, 0.0, 0.0, 0.1405,
//                0.0, -1.0, 0.0, 0.1485,
//                0.0, 0.0, 1.0, 0.2642,
//                0.0, 0.0, 0.0, 1.0;



class WheelDataAnalysis : public ParamServer
{

public:
    rclcpp::Subscription<gpal_msgs::msg::VcuData>::SharedPtr sub_vcu_data_;
    rclcpp::Subscription<gpal_msgs::msg::VcuOdom>::SharedPtr sub_vcu_odom_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_vcu_nav_odom_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gps_data_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_data_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vcuodom_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vcudata_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vcu_navodom_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_vcu_imu_path_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_vcu_imu_odom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_gps_path_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;

    nav_msgs::msg::Path vcu_odom_path_;
    nav_msgs::msg::Path vcu_data_path_;
    nav_msgs::msg::Path vcu_nav_odom_path_;
    std::deque<gpal_msgs::msg::VcuData> wheelOdomQueue;

    GeographicLib::LocalCartesian geo_converter;

    message_filters::Subscriber<nav_msgs::msg::Odometry> vcu_odom_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Imu> imu_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<nav_msgs::msg::Odometry, sensor_msgs::msg::Imu>> sync_;

    std::ofstream gps_ofs_, vcu_data_ofs_, vcu_nav_odom_ofs_;
    std::ofstream vcu_imu_ofs_;

    Eigen::Vector3d curr_acc = Eigen::Vector3d::Identity();
    double curr_angular = 0;
    std::mutex imu_acc_mux;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;

    pcl::PointCloud<PointType>::Ptr vcuOdomPointCloud;

    WheelDataAnalysis(const rclcpp::NodeOptions &options) : ParamServer("wheel", options)
    {
        // sub_vcu_data_ = create_subscription<gpal_msgs::msg::VcuData>(
        //     vcu_data, 1,
        //     std::bind(&WheelDataAnalysis::vcuDataHandler, this, std::placeholders::_1));

        // sub_vcu_odom_ = create_subscription<gpal_msgs::msg::VcuOdom>(
        //     vcu_odom, 1,
        //     std::bind(&WheelDataAnalysis::vcuOdomHandler, this, std::placeholders::_1));

        // sub_vcu_nav_odom_ = create_subscription<nav_msgs::msg::Odometry>(sub_vcu_odom_ = create_subscription<gpal_msgs::msg::VcuOdom>(
        //     vcu_odom, 1,
        //     std::bind(&WheelDataAnalysis::vcuOdomHandler, this, std::placeholders::_1));

        sub_vcu_nav_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            vcu_nav_odom, qos,
            std::bind(&WheelDataAnalysis::vcuNavOdomHandler, this, std::placeholders::_1));
        
        auto imuOpt = rclcpp::SubscriptionOptions();
        callbackGroupImu = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        imuOpt.callback_group = callbackGroupImu;
        sub_imu_data_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_data, qos_imu,
            std::bind(&WheelDataAnalysis::ImuHandler, this, std::placeholders::_1), imuOpt);

        // sub_gps_data_ = create_subscription<sensor_msgs::msg::NavSatFix>(gps_data, 1, std::bind(&WheelDataAnalysis::gpsHandler, this, std::placeholders::_1));

        // vcu_odom_sub_.subscribe(this, vcu_nav_odom);
        // imu_sub_.subscribe(this, "/imu");
        // sync_ = std::make_shared<message_filters::TimeSynchronizer<nav_msgs::msg::Odometry, sensor_msgs::msg::Imu>>(vcu_odom_sub_, imu_sub_, 3);
        // sync_->registerCallback(std::bind(&WheelDataAnalysis::vcuOdomImuSync, this, std::placeholders::_1, std::placeholders::_2));

        // pub_vcuodom_path_ = create_publisher<nav_msgs::msg::Path>("vcuodom_wheel_path", 100);
        // pub_vcudata_path_ = create_publisher<nav_msgs::msg::Path>("vcudata_wheel_path", 100);
        pub_vcu_navodom_path_ = create_publisher<nav_msgs::msg::Path>("vcu_navodom_wheel_path", 100);
        // pub_vcu_imu_path_ = create_publisher<nav_msgs::msg::Path>("vcu_imu_path", 100);

        // pub_gps_path_ = create_publisher<nav_msgs::msg::Path>("gps_path", 100);
        pub_vcu_imu_odom_ = create_publisher<nav_msgs::msg::Odometry>("vcu_imu_odom", 100);

        // gps_ofs_.open("gps.txt", std::ios::out | std::ios::app);
        // vcu_data_ofs_.open("vcu_data.txt", std::ios::out | std::ios::app);
        vcu_nav_odom_ofs_.open("vcu_nav_odom.txt", std::ios::out | std::ios::app);
        // vcu_imu_ofs_.open("vcu_imu.txt", std::ios::out | std::ios::app);

        vcuOdomPointCloud.reset(new pcl::PointCloud<PointType>());
    }

    void ImuHandler(const sensor_msgs::msg::Imu::SharedPtr imu)
    {
        //std::lock_guard<std::mutex> lock(imu_acc_mux);
        curr_acc << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
        curr_angular = -imu->angular_velocity.z;
        return;
    }

    void vcuOdomImuSync(const nav_msgs::msg::Odometry::ConstSharedPtr &vcu_odom, const sensor_msgs::msg::Imu::ConstSharedPtr &imu)
    {
        static double last_time = 0;
        static bool first_flag = true;
        static double x, y, yaw = 0;
        static nav_msgs::msg::Odometry preOdom;
        nav_msgs::msg::Odometry curOdom = *vcu_odom;
        if (first_flag)
        {
            last_time = rclcpp::Time(vcu_odom->header.stamp).seconds();
            first_flag = false;
            preOdom = *vcu_odom;
            return;
        }
       
        double cur_time = rclcpp::Time(vcu_odom->header.stamp).seconds();
        double dt = cur_time - last_time;
        // curOdom = *vcu_odom;
        curOdom.header.frame_id = "odom";
        curOdom.child_frame_id = "base_link";
        // tf2::Quaternion q;
        // tf2::convert(preOdom.pose.pose.orientation, q);
        // double roll1, pitch1, yaw1;
        // tf2::Matrix3x3(q).getRPY(roll1, pitch1, yaw1);

        // tf2::convert(curOdom.pose.pose.orientation, q);
        // double roll2, pitch2, yaw2;
        // tf2::Matrix3x3(q).getRPY(roll2, pitch2, yaw2);
        // double w = curOdom.twist.twist.angular.z;
        double w = imu->angular_velocity.z > 0 ? imu->angular_velocity.z : -imu->angular_velocity.z;
        yaw += w * dt;
        if (yaw > 2 * M_PI)
            yaw = 2 * M_PI - yaw;
        else if (yaw < -2 * M_PI)
            yaw += 2 * M_PI;
        // double yaw = (180 / M_PI) * yaw1;
        // static double yaww = 0;
        // yaww += yaw;
        // double cur_speed = (preOdom.twist.twist.linear.x + curOdom.twist.twist.linear.x) / 2;
        double cur_speed = curOdom.twist.twist.linear.x;

        double delta_x = cur_speed * cos(yaw) * dt;
        double delta_y = cur_speed * sin(yaw) * dt;
        x += delta_x;
        y += delta_y;

        // path
        static nav_msgs::msg::Path vcu_odom_path;
        vcu_odom_path.header.frame_id = "odom";
        vcu_odom_path.header.stamp = curOdom.header.stamp;
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "odom";
        pose.header.stamp = curOdom.header.stamp;
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 0;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        // pose.pose.orientation = preOdom.pose.pose.orientation;
        vcu_odom_path.poses.push_back(pose);
        pub_vcu_imu_path_->publish(vcu_odom_path);

        last_time = cur_time;
        preOdom = curOdom;

        // vcu_imu_ofs_ << std::fixed << std::setprecision(6) << rclcpp::Time(vcu_odom->header.stamp).seconds() << " "
        //              << x << " " << y << " " << 0.0 << " "
        //              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = "odom";
        odom.header.stamp = curOdom.header.stamp;
        odom.pose.pose = pose.pose;
        // pub_vcu_imu_odom_->publish(odom);

    }

    void vcuDataHandler(const gpal_msgs::msg::VcuData::SharedPtr msg)
    {
        gpal_msgs::msg::VcuData cur_data = *msg;
        static gpal_msgs::msg::VcuData pre_data;
        static bool first_frame = true;
        if (first_frame)
        {
            pre_data = cur_data;
            first_frame = false;
            return;
        }
        static double x, y, yaw = 0;
        int model = cur_data.drive_mode;
        if (model == 2) // 倒车档
        {
            cur_data.speed = -cur_data.speed;
        }
        else if (model == 1) // 停车档
        {
            cur_data.speed = 0;
        }

        double dt = rclcpp::Time(cur_data.header.stamp).seconds() - rclcpp::Time(pre_data.header.stamp).seconds();
        std::cout << "dt: " << dt << std::endl;
        // 计算当前帧与上一帧的平均speed与角速度
        // double odom_speed = pre_data.speed;
        // double odom_yaw_rate = pre_data.yaw_rate;

        // double rear_odom_speed = (cur_data.rear_right_wheel_speed + cur_data.rear_left_wheel_speed) / 2;
        double speed = cur_data.speed;
        // 输出dyaw dx dy
        double dyaw = cur_data.yaw_rate;
        double w = (cur_data.rear_right_wheel_speed - cur_data.rear_left_wheel_speed) / L_REAR;
        // std::cout << "w1:" << w << " w2:" << dyaw << " e_w:" << abs(dyaw - w) << std::endl;
        double dx, dy = 0.0;
        // yaw += dyaw * dt;
        yaw += w * dt;
        if (yaw > 2 * M_PI)
            yaw = 2 * M_PI - yaw;
        else if (yaw < -2 * M_PI)
            yaw = yaw + 2 * M_PI;
        dx = dt * speed * cos(yaw);
        dy = dt * speed * sin(yaw);

        x += dx;
        y += dy;
        // 发布path
        // std ::cout << "vcu_data==: " <<" dx: " << dx << " dy: " << dy << " dyaw: " << dyaw << std::endl;
        // std ::cout << "vcu_data==: " <<"x: " << x << " y: " << y << " yaw: " << yaw << std::endl;
        geometry_msgs::msg::PoseStamped pose;
        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = 0;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        pose.header.frame_id = "odom";
        vcu_data_path_.header = pose.header;
        vcu_data_path_.poses.push_back(pose);
        pub_vcudata_path_->publish(vcu_data_path_);
        pre_data = cur_data;
        //}

        // wheelOdomQueue.push_back(cur_data);
        //}
        // count++;

        // // 控制队列大小
        // if (wheelOdomQueue.size() > 4)
        // {
        //     wheelOdomQueue.pop_front();
        // }
        vcu_data_ofs_ << std::fixed << std::setprecision(6) << rclcpp::Time(cur_data.header.stamp).seconds() << " "
                      << x << " " << y << " " << 0.0 << " "
                      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    void vcuOdomHandler(const gpal_msgs::msg::VcuOdom::SharedPtr msg)
    {
        gpal_msgs::msg::VcuOdom cur_odom = *msg;

        // 当前帧与上一帧相对变换
        static double x, y, yaw = 0;
        x += cur_odom.dx;
        y += cur_odom.dy;
        yaw += cur_odom.dyaw;
        // std::cout << "vcu_odom==: " << "x: " << x << " y: " << y << " yaw: " << yaw << std::endl;
        // 发布path
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = cur_odom.header.stamp;
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = x;
        pose_stamped.pose.position.y = y;
        pose_stamped.pose.position.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
        vcu_odom_path_.header = pose_stamped.header;
        vcu_odom_path_.poses.push_back(pose_stamped);
        pub_vcuodom_path_->publish(vcu_odom_path_);
    }

    void vcuNavOdomHandler(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        static double last_time = 0;
        static bool first_flag = true;
        static double x, y, yaw = 0;
        static nav_msgs::msg::Odometry preOdom;
        static nav_msgs::msg::Odometry curOdom;
        if (first_flag)
        {
            last_time = rclcpp::Time(msg->header.stamp).seconds();
            first_flag = false;
            preOdom = *msg;
            return;
        }
        // static int count = 0;
        // if (count % 50 == 0)
        // {
        double cur_time = rclcpp::Time(msg->header.stamp).seconds();
        double dt = cur_time - last_time;
        curOdom = *msg;
        curOdom.header.frame_id = "odom";
        curOdom.child_frame_id = "base_link";
        // tf2::Quaternion q;
        // tf2::convert(preOdom.pose.pose.orientation, q);
        // double roll1, pitch1, yaw1;
        // tf2::Matrix3x3(q).getRPY(roll1, pitch1, yaw1);

        // tf2::convert(curOdom.pose.pose.orientation, q);
        // double roll2, pitch2, yaw2;
        // tf2::Matrix3x3(q).getRPY(roll2, pitch2, yaw2);

        double w = curOdom.twist.twist.angular.z;
        //cout << "w:" << w << "imu:" << curr_angular << endl;
        yaw += curr_angular * dt;
        if (yaw > M_PI)
            yaw -= 2 * M_PI;
        else if (yaw < -M_PI)
            yaw += 2 * M_PI;
        // double yaw = (180 / M_PI) * yaw1;
        // static double yaww = 0;
        // yaww += yaw;
        // double cur_speed = (preOdom.twist.twist.linear.x + curOdom.twist.twist.linear.x) / 2;
        double cur_speed = curOdom.twist.twist.linear.x;

        double delta_x = cur_speed * cos(yaw) * dt;
        double delta_y = cur_speed * sin(yaw) * dt;
        x += delta_x;
        y += delta_y;

        last_time = cur_time;
        preOdom = curOdom;

        tf2::Transform trans_imu;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        trans_imu.setOrigin(tf2::Vector3(x, y, 0.0));
        trans_imu.setRotation(q);

        tf2::Transform trans_imu_base;
        //trans_imu_base.setOrigin(tf2::Vector3(0.1405, 0.1485, 0.2642));
        tf2::Quaternion q_imu_base;
        q_imu_base.setRPY(M_PI, 0, 0);
        trans_imu_base.setRotation(q_imu_base);

        // tf2::Transform trans_base =  trans_imu_base * trans_imu;
        tf2::Transform trans_base = trans_imu;

        // path
        static nav_msgs::msg::Path vcu_odom_path_;
        vcu_odom_path_.header.frame_id = "odom";
        vcu_odom_path_.header.stamp = curOdom.header.stamp;
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "odom";
        pose.header.stamp = curOdom.header.stamp;
        pose.pose.position.x = trans_base.getOrigin().x();
        pose.pose.position.y = trans_base.getOrigin().y();
        pose.pose.position.z = 0;

        pose.pose.orientation.x = trans_base.getRotation().x();
        pose.pose.orientation.y = trans_base.getRotation().y();
        pose.pose.orientation.z = trans_base.getRotation().z();
        pose.pose.orientation.w = trans_base.getRotation().w();
        // std::cout << trans_base.getRotation().x() << " " << trans_base.getRotation().y() << " "
        //           << trans_base.getRotation().z() << " " << trans_base.getRotation().w() << std::endl;
        // pose.pose.orientation = preOdom.pose.pose.orientation;
        vcu_odom_path_.poses.push_back(pose);
        pub_vcu_navodom_path_->publish(vcu_odom_path_);

        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = "odom";
        odom.header.stamp = curOdom.header.stamp;
        odom.pose.pose = pose.pose;
        pub_vcu_imu_odom_->publish(odom);

        
        // ofsVcuPath.open("/home/yaya/SensorTestRos2/trajectory_vcu.txt", std::ofstream::out | std::ofstream::app);
        // ofsVcuPath << last_time << "," << x << "," << y << "," << 0 << "," << pose.pose.orientation.x << "," << pose.pose.orientation.y << "," << pose.pose.orientation.z << "," << pose.pose.orientation.w << std::endl;
        // ofsVcuPath.close();
        // }
        // count++;
        vcu_nav_odom_ofs_ << std::fixed << std::setprecision(6) << rclcpp::Time(curOdom.header.stamp).seconds() << " "
                          << x << " " << y << " " << 0.0 << " "
                          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    void gpsHandler(const sensor_msgs::msg::NavSatFix::SharedPtr gps_msg)
    {
        static bool first_keyframe = true;
        if (first_keyframe)
        {
            geo_converter.Reset(gps_msg->latitude, gps_msg->longitude, gps_msg->altitude);
            first_keyframe = false;
        }
        double x, y, z;
        geo_converter.Forward(gps_msg->latitude, gps_msg->longitude, gps_msg->altitude, x, y, z);

        geometry_msgs::msg::PoseStamped gps_p;
        gps_p.header.stamp = gps_msg->header.stamp;
        gps_p.pose.position.x = x;
        gps_p.pose.position.y = y;
        gps_p.pose.position.z = z;

        // 发布gps轨迹
        static nav_msgs::msg::Path gpsPath;
        gpsPath.poses.push_back(gps_p);
        gpsPath.header.frame_id = "odom";
        gpsPath.header.stamp = gps_p.header.stamp;
        pub_gps_path_->publish(gpsPath);
        // // 写入txt文件
        // ofsGpsPath.open("/home/yaya/SensorTestRos2/trajectory_gps.txt",std::ofstream::out | std::ofstream::app);
        // ofsGpsPath << rclcpp::Time(gps_msg->header.stamp).seconds() << "," <<
        //             x << "," << y << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 1 << std::endl;
        // ofsGpsPath.close();
        // gps_ofs_ << std::fixed << std::setprecision(6) << rclcpp::Time(gps_msg->header.stamp).seconds() << " "
        //          << x << " " << y << " " << 0.0 << " "
        //          << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<WheelDataAnalysis>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> wheel Data Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}