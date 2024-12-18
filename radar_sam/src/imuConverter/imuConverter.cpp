
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


class imuConvert : public ParamServer
{

public:
   
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_data_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;

    imuConvert(const rclcpp::NodeOptions &options) : ParamServer("imuConverter", options)
    {
        auto imuOpt = rclcpp::SubscriptionOptions();
        callbackGroupImu = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        imuOpt.callback_group = callbackGroupImu;
        sub_imu_data_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu_raw", qos_imu,
            std::bind(&imuConvert::ImuHandler, this, std::placeholders::_1), imuOpt);
        pub_imu_ =  create_publisher<sensor_msgs::msg::Imu>("imu", 100);
    }

    void ImuHandler(const sensor_msgs::msg::Imu::SharedPtr imu)
    {
        sensor_msgs::msg::Imu thisImu = imuConverter(*imu);
        pub_imu_->publish(thisImu);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<imuConvert>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU converter Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}