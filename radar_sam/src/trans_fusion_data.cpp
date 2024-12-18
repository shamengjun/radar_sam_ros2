#include "radar_sam/msg/fusion_data_frame.hpp"
#include "radar_sam/msg/fusion_points.hpp"
#include "radar_sam/msg/odom_msg.hpp"
#include "utility.h"

using std::cout;
using std::endl;

#define RADAR_DEBUG 0
class TransFusionData : public ParamServer
{
public:

rclcpp::Subscription<radar_sam::msg::FusionDataFrame>::SharedPtr subFusionData;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubFusionPoints;

   TransFusionData(const rclcpp::NodeOptions & options) :
        ParamServer("radar_point_prefilter", options)
   {
 
      subFusionData = create_subscription<radar_sam::msg::FusionDataFrame>(
            m_fusion_topic_name, qos_lidar,
            std::bind(&TransFusionData::radarCloudHandler, this, std::placeholders::_1));

      pubFusionPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar);
   }


   void radarCloudHandler(const radar_sam::msg::FusionDataFrame::SharedPtr msgIn)
   {
      radar_sam::msg::FusionDataFrame radarCloudPoint = *msgIn;
      // 暂定只接收xyz
      pcl::PointCloud<pcl::PointXYZ> allPointsXYZ;
      std::map<int, int> radarPointsMap; //统计各个radar的点数

#pragma omp parallel for num_threads(numberOfCores)
      // 看一下耗时
      auto start = std::chrono::high_resolution_clock::now();
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
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << "For循环耗时: " << duration << " 毫秒" << std::endl;
      // 发布
      sensor_msgs::msg::PointCloud2 radarCloudPoints;
      allPointsXYZ.width = allPointsXYZ.points.size();
      allPointsXYZ.height = 1;
      pcl::toROSMsg(allPointsXYZ, radarCloudPoints);
      radarCloudPoints.header.frame_id = "radar_link";
      radarCloudPoints.header.stamp = rclcpp::Clock().now();
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

