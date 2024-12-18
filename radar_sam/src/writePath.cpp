#include "utility.h"
#include <Eigen/Dense> //四元数转旋转向量
#include "radar_sam/srv/save_map.hpp"

struct PointXYZIRPYTLLA
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    double longitude;
    double latitude;
    double altitude;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYTLLA,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                  (float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)
                                  (double, time, time)
                                  (double, longitude, longitude)(double, latitude, latitude)(double, altitude, altitude))

typedef PointXYZIRPYTLLA PointTypePose;
using namespace std;

class writePath : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;       // sub gps+imu filterd path
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subFilterdGps; // sub gps+imu filterd path

    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubFilteredPath;
    rclcpp::Publisher<gpal_vision_msgs::msg::OverlayText>::SharedPtr pubOverlayText;

    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses3D; // 用来存所有状态优化后的值

    ofstream write_path_ofs;

    string save_file_name_prefix;

    Eigen::Vector3d curr_gps;

    pcl::PointCloud<PointTypePose>::Ptr filtered_pose;
    
    rclcpp::Service<radar_sam::srv::SaveMap>::SharedPtr save_path_srv;

    rclcpp::TimerBase::SharedPtr save_path_timer;

    writePath(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_vcu_gps_optimization", options)
    {

        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "/odometry/filtered", qos,
            std::bind(&writePath::odometryHandler, this, std::placeholders::_1));

        subFilterdGps = create_subscription<sensor_msgs::msg::NavSatFix>(
            "/gps/filtered", qos,
            std::bind(&writePath::gpsHandler, this, std::placeholders::_1));

        // odomTopic -> odometry/imu
        pubFilteredPath = create_publisher<nav_msgs::msg::Path>("filtered_path", qos_imu);

        pubOverlayText = create_publisher<gpal_vision_msgs::msg::OverlayText>("state_overlay", qos);

       
        rclcpp::CallbackGroup::SharedPtr group = create_callback_group(
                                                    rclcpp::CallbackGroupType::MutuallyExclusive);
        save_path_srv = create_service<radar_sam::srv::SaveMap>("save_path", 
                        std::bind(&writePath::savePathSrv, this, std::placeholders::_1, std::placeholders::_2));
        
        save_path_timer = create_wall_timer(5 * 60 *1s, std::bind(&writePath::savePathTimerCB, this)); // 

        
        save_file_name_prefix = "imu_gps_filtered-" + getCurrentTime();
    
        write_path_ofs.open(save_file_name_prefix + ".tum", std::ios::out | std::ios::app);
        write_path_ofs << "#timestamp  x y z qx qy qz qw f_lon f_lat f_ati" << std::endl;

        filtered_pose.reset(new pcl::PointCloud<PointTypePose>());
    }

    ~writePath()
    {
        // 保存pcd
        string pcd_file_name = save_file_name_prefix + ".pcd";
        pcl::io::savePCDFileBinary(pcd_file_name, *filtered_pose);
    }

    void savePathTimerCB()
    {
        static int cnt = 0;
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "ros2 timer cnt:%d", cnt++);
        return;
    }

    bool savePathSrv(const std::shared_ptr<radar_sam::srv::SaveMap::Request> req, 
                     const std::shared_ptr<radar_sam::srv::SaveMap::Response> res)
    {
        string save_name;
        if (req->resolution != 0)
        {
            pcl::VoxelGrid<PointTypePose> downSizeFilter;
            downSizeFilter.setInputCloud(filtered_pose);
           
            downSizeFilter.setLeafSize(req->resolution, req->resolution, req->resolution);
            downSizeFilter.filter(*filtered_pose);
        }

        if (req->destination != " ")
        {
            save_name = req->destination;
        }
        else
           save_name = save_file_name_prefix + ".pcd";
        
        pcl::io::savePCDFileBinary(save_name, *filtered_pose);

        res->success = true;
        return true;
    }

    string getCurrentTime()
    {
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        char buf[100] = {0};
        std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&now));
        return buf;
    }

    double disBetweenGpsOdom(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
    {
        return std::sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                         (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                         (p1[2] - p2[2]) * (p1[2] - p2[2]));
    }

    double poseDistance(PointTypePose p1, PointTypePose p2)
    {
        return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
    }

    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        double currentTime = stamp2Sec(odomMsg->header.stamp); // odom的当前时间

        // 发布path
        static nav_msgs::msg::Path curr_opt_path;
        static double last_time = -1;
        double curr_vel = 0;
        static double odometry = 0.0;
        static double last_x = 0.0;
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = odomMsg->header.stamp;
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = odomMsg->pose.pose.position.x;
        pose_stamped.pose.position.y = odomMsg->pose.pose.position.y;
        pose_stamped.pose.position.z = odomMsg->pose.pose.position.z;
        pose_stamped.pose.orientation.x = odomMsg->pose.pose.orientation.x;
        pose_stamped.pose.orientation.y = odomMsg->pose.pose.orientation.y;
        pose_stamped.pose.orientation.z = odomMsg->pose.pose.orientation.z;
        pose_stamped.pose.orientation.w = odomMsg->pose.pose.orientation.w;
        if (!curr_opt_path.poses.empty())
        {
            double diff_x = odomMsg->pose.pose.position.x - curr_opt_path.poses.back().pose.position.x;
            double diff_y = odomMsg->pose.pose.position.x - curr_opt_path.poses.back().pose.position.y;
            odometry += sqrt(diff_x * diff_x + diff_y * diff_y);
            curr_vel = sqrt(diff_x * diff_x + diff_y * diff_y) / (currentTime - last_time);
        }
        curr_opt_path.poses.push_back(pose_stamped);

        last_time = currentTime;
        if (pubFilteredPath->get_subscription_count() != 0)
        {
            curr_opt_path.header.stamp = odomMsg->header.stamp;
            curr_opt_path.header.frame_id = odometryFrame;
            pubFilteredPath->publish(curr_opt_path);
        }

        if (pubOverlayText->get_subscription_count() != 0)
        {

            // double curr_angular_yaw = prevState_.angular().z();
            std::string overlay_string = "[opt_vel]:" + std::to_string(curr_vel * 3.6) + "km/h  " + +"[odometry]:" + std::to_string(odometry / 1000) + "km";
            publishOverlayText(overlay_string, pubOverlayText, "r");
        }

        static int frame_cnt = 0;
        if (!((frame_cnt++) % 10 == 0))
            return;

        write_path_ofs << std::fixed << std::setprecision(8) << currentTime << " "
                       << odomMsg->pose.pose.position.x << " " << odomMsg->pose.pose.position.y << " " << odomMsg->pose.pose.position.z << " "
                       << odomMsg->pose.pose.orientation.x << " " << odomMsg->pose.pose.orientation.y << " "
                       << odomMsg->pose.pose.orientation.z << " " << odomMsg->pose.pose.orientation.w << " "
                       << curr_gps[0] << " " << curr_gps[1] << " " << curr_gps[2] << std::endl;

        PointTypePose p;
       
        tf2::Quaternion orientation;
        tf2::fromMsg(odomMsg->pose.pose.orientation, orientation);

        // 获得此时rpy
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        p.x = odomMsg->pose.pose.position.x;
        p.y = odomMsg->pose.pose.position.y;
        p.z = odomMsg->pose.pose.position.z;
        p.roll = roll;
        p.pitch = pitch;
        p.yaw = yaw;
        p.time = currentTime;
        p.longitude = curr_gps[0];
        p.latitude = curr_gps[1];
        p.altitude = curr_gps[2];
        filtered_pose->points.push_back(p);
    }

    void gpsHandler(const sensor_msgs::msg::NavSatFix::SharedPtr gpsOdom)
    {
        std::lock_guard<std::mutex> lock(mutex);
        curr_gps << gpsOdom->latitude, gpsOdom->longitude, gpsOdom->altitude;
    }
};

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto vcuGpsOpt = std::make_shared<writePath>(options);

    exec.add_node(vcuGpsOpt);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> writePath Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();

    return 0;
}
