#include "radar_sam/msg/fusion_data_frame.hpp"
#include "radar_sam/msg/fusion_points.hpp"
#include "radar_sam/msg/odom_msg.hpp"
#include "utility.h"
#include "radar_msgs/msg/radar_target.hpp"
#include "gpal_msgs/msg/fusion_points.hpp"
#include "gpal_msgs/msg/fusion_data_frame.hpp"
#include "gpal_msgs/msg/vcu_data.hpp"

using std::cout;
using std::endl;

#define RADAR_DEBUG 0
class TransFusionData : public ParamServer
{
public:

rclcpp::Subscription<gpal_msgs::msg::FusionDataFrame>::SharedPtr subFusionData;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubFusionPoints;

   TransFusionData(const rclcpp::NodeOptions & options) :
        ParamServer("radar_point_prefilter", options)
   {
 
      subFusionData = create_subscription<gpal_msgs::msg::FusionDataFrame>(
            m_fusion_topic_name, qos_lidar,
            std::bind(&TransFusionData::radarCloudHandler, this, std::placeholders::_1));

      pubFusionPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar);
   }


   void radarCloudHandler(const gpal_msgs::msg::FusionDataFrame::SharedPtr msgIn)
   {
      // std::cout << "===========radarCloudHandler===========" << std::endl;
      gpal_msgs::msg::FusionDataFrame radarCloudPoint = *msgIn;
      gpal_msgs::msg::FusionDataFrame* radarCloudPointPt = &radarCloudPoint;
      // 暂定只接收xyz
      // std::cout << "radarCloudPoint.header.stamp: " << stamp2Sec(radarCloudPoint.header.stamp) << std::endl;
      // std::cout << "radarCloudPoint.header.stamp.sec " << radarCloudPointPt->header.stamp.sec <<" " << radarCloudPointPt->header.stamp.nanosec << std::endl;
      pcl::PointCloud<pcl::PointXYZ> allPointsXYZ;
      std::map<int, int> radarPointsMap; //统计各个radar的点数

#pragma omp parallel for num_threads(numberOfCores)
      // 看一下耗时
      std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
      // 打一下队列大小
      // std::cout << "queue size: " << radarCloudPoint.radar_points.size() << std::endl;
      for (int i = 0; i < radarCloudPoint.radar_points.size(); i++)
      {
         radarPointsMap[radarCloudPoint.radar_points[i].radar_id]++;
         if (radarCloudPoint.radar_points[i].motion_state) //动态点删掉
            continue;
         pcl::PointXYZ point;
         point.x = radarCloudPoint.radar_points[i].x;
         point.y = radarCloudPoint.radar_points[i].y;
         point.z = radarCloudPoint.radar_points[i].z;
         allPointsXYZ.points.push_back(point);
      }
      std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      // std::cout << "代码块执行时间: " << elapsed_time.count() << " 秒" << std::endl;
      // 发布
      sensor_msgs::msg::PointCloud2 radarCloudPoints;
      allPointsXYZ.width = allPointsXYZ.points.size();
      allPointsXYZ.height = 1;
      pcl::toROSMsg(allPointsXYZ, radarCloudPoints);
      radarCloudPoints.header.frame_id = "radar_link";
      radarCloudPoints.header.stamp = msgIn->header.stamp;
      pubFusionPoints->publish(radarCloudPoints);
   }

};

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<TransFusionData>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> TransFusionData Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}

