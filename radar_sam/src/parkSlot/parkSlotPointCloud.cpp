
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include "gpal_msgs/msg/vcu_data.hpp"
#include "gpal_msgs/msg/vcu_odom.hpp"
#include "gpal_msgs/msg/ctl_p_msgs_array.hpp"
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

/*
  将视觉语义的消息转化为点云的格式
*/

class parkSlotPointCloud : public ParamServer
{
public:
   
    rclcpp::Subscription<gpal_msgs::msg::CtlPMsgsArray>::SharedPtr sub_park_semantic_data_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_points;
   

    parkSlotPointCloud(const rclcpp::NodeOptions &options) : ParamServer("parkSlotPointCloud", options)
    {
        sub_park_semantic_data_ = create_subscription<gpal_msgs::msg::CtlPMsgsArray>("track_park", qos, 
            std::bind(&parkSlotPointCloud::parkSemanticSubCB, this, std::placeholders::_1));
        pub_semantic_points = create_publisher<sensor_msgs::msg::PointCloud2>("/park_points", qos);
    }

    void parkSemanticSubCB(const gpal_msgs::msg::CtlPMsgsArray::SharedPtr parkMsg)
    {
        if (parkMsg->ctlparray.empty())
          return;
        //RCLCPP_INFO(get_logger(), "receive parkMsg.");
        SenmanticsInfoPointType park_semantics;
        for (int i = 0; i < parkMsg->ctlparray.size(); ++i)
        {
            XYZRGBSemanticsInfo pa;
            XYZRGBSemanticsInfo pb;
            XYZRGBSemanticsInfo pc;
            XYZRGBSemanticsInfo pd;
            pa.x = parkMsg->ctlparray[i].pax;
            pa.y = parkMsg->ctlparray[i].pay;
            pa.z = 1.0;

            pb.x = parkMsg->ctlparray[i].pbx;
            pb.y = parkMsg->ctlparray[i].pby;
            pb.z = 1.0;

            pc.x = parkMsg->ctlparray[i].pcx;
            pc.y = parkMsg->ctlparray[i].pcy;
            pc.z = 1.0;

            pd.x = parkMsg->ctlparray[i].pdx;
            pd.y = parkMsg->ctlparray[i].pdy;
            pd.z = 1.0;

            pa.id = parkMsg->ctlparray[i].ids;
            pb.id = parkMsg->ctlparray[i].ids;
            pc.id = parkMsg->ctlparray[i].ids;
            pd.id = parkMsg->ctlparray[i].ids;

            if (parkMsg->ctlparray[i].cc > 0) //占据
            {
               pa.r = 255;
               pa.g = 0;
               pa.b = 0; 

               pb.r = 255;
               pb.g = 0;
               pb.b = 0;

               pc.r = 255;
               pc.g = 0;
               pc.b = 0;

               pd.r = 255;
               pd.g = 0;
               pd.b = 0;
            }
            else
            {
               pa.r = 0;
               pa.g = 255;
               pa.b = 0;  
               
               pb.r = 0;
               pb.g = 255;
               pb.b = 0; 

               pc.r = 0;
               pc.g = 255;
               pc.b = 0; 

               pd.r = 0;
               pd.g = 255;
               pd.b = 0; 
            }
            pa.score = parkMsg->ctlparray[i].score;
            pa.t = rclcpp::Time(parkMsg->header.stamp).seconds();
            
            pb.score = parkMsg->ctlparray[i].score;
            pb.t = rclcpp::Time(parkMsg->header.stamp).seconds();

            pc.score = parkMsg->ctlparray[i].score;
            pc.t = rclcpp::Time(parkMsg->header.stamp).seconds();

            pd.score = parkMsg->ctlparray[i].score;
            pd.t = rclcpp::Time(parkMsg->header.stamp).seconds();

            park_semantics.points.push_back(pa);
            park_semantics.points.push_back(pb);
            park_semantics.points.push_back(pc);
            park_semantics.points.push_back(pd);
        }

        //转化为ros发布出来
        sensor_msgs::msg::PointCloud2 park_points;
        pcl::toROSMsg(park_semantics, park_points);
        park_points.header.stamp = parkMsg->header.stamp;
        park_points.header.frame_id = "base_link";
        pub_semantic_points->publish(park_points);
        //RCLCPP_INFO(get_logger(), "send park points.");
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<parkSlotPointCloud>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m---->parkslot point clouds Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}