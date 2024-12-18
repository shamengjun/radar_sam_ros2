#include "utility.h"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include "gpal_msgs/msg/vcu_data.hpp"
#include <pcl/filters/extract_indices.h>
#include "gpal_msgs/msg/ctl_p_msgs_array.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <algorithm>
#include "../tinyxml2/tinyxml2.h"

using namespace std;
using namespace tinyxml2;
// 以下的pointType不是点云的类型 是位姿Pose的类型
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  double his_time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time)(double, his_time, his_time))

// 6自由度的位姿和其对应的时间
typedef PointXYZIRPYT PointTypePose;

struct slotInfo
{
  int id;
  double score;
  int cc;
  std::vector<pcl::PointXYZ> corner_points;
  int keyframe_id; // 标识在哪个关键帧下的建立的车位
  // 其余的暂且不关心，后续有需要再定义
  double time; // 啥时候检出
  double center_x;
  double center_y; // 中心点位置用来可视化ID

  bool operator==(const slotInfo &slot)
  {
    return slot.id == id;
  }
};

// bool is_remove(int id1, int id2)
// {
//   return id2 == id2;
// }

class PointsFusion : public ParamServer
{

public:
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRadarCloudSurround;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubSlotInfo;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubAllSlotInfo;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subPointCloud;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subPose;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr finish_mapping_sub;
  rclcpp::Subscription<gpal_msgs::msg::VcuData>::SharedPtr subVcuData;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subGPSRaw;
  rclcpp::Subscription<gpal_msgs::msg::CtlPMsgsArray>::SharedPtr subSlot;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subPath;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr subCorrectMsg;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr subSaveSlotsMsg;

  double timeRadarInfoCur;
  rclcpp::Time timeRadarInfoStamp;
  double transformTobeMapped[6];
  pcl::VoxelGrid<PointType> downSizeFilterCorner;
  nav_msgs::msg::Odometry curPose;
  std::deque<nav_msgs::msg::Odometry> poseQueue;
  std::deque<gpal_msgs::msg::VcuData> vcuDataQueue;
  std::deque<sensor_msgs::msg::NavSatFix> gpsRawQueue;
  std::map<int, std::vector<double>> gps_key_map;
  pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr radarCloudRaw;
  pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr pointCloudAll;
  pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
  // vector<pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr> allCloudKeyFrames;
  vector<std::list<slotInfo>> allSlotsKeyFrames;
  std::list<slotInfo> currSlots;
  std::vector<double> keyFramePoseTimestamp;
  std::vector<int> vcu_gear;
  std::mutex mtx;
  std::mutex vcuMtx;
  std::mutex mtxGPS;
  // std::mutex poseMtx;
  std::ofstream save_trajectory;
  std::ofstream save_tum_pose;

  std::map<int, std::vector<int>> slotsIds;

  pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
  pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr laserCloudAllFromMap;

  bool isCorrectPose;

  PointsFusion(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_pointsFusion", options)
  {

    pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/local_map", 1);
    pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/cloud_registered", 1);
    pubRadarCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/map_global", 1);
    pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("radar_sam/point_fusion/trajectory", 1);
    pubSlotInfo = create_publisher<visualization_msgs::msg::MarkerArray>("/parking_space", 1);
    pubAllSlotInfo = create_publisher<visualization_msgs::msg::MarkerArray>("/all_slots", 1);
    // subPointCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
    //     "park_points", qos_lidar,
    //     std::bind(&PointsFusion::pointCloudInfoHandler, this, std::placeholders::_1));
    subSlot = create_subscription<gpal_msgs::msg::CtlPMsgsArray>("track_park", qos,
                                                                 std::bind(&PointsFusion::parkSemanticSubCB, this, std::placeholders::_1));

    subPose = create_subscription<nav_msgs::msg::Odometry>(
        "opt_odom", qos_imu,
        std::bind(&PointsFusion::poseInfoHandler, this, std::placeholders::_1));

    subPath = create_subscription<nav_msgs::msg::Path>(
        "opt_path", qos_imu,
        std::bind(&PointsFusion::pathSubHandler, this, std::placeholders::_1));

    subCorrectMsg = create_subscription<std_msgs::msg::Empty>(
        "correct_pose", qos, std::bind(&PointsFusion::correctPoseSub, this, std::placeholders::_1));

    subSaveSlotsMsg = create_subscription<std_msgs::msg::Empty>(
        "save_all_slots", qos, std::bind(&PointsFusion::saveSlotsSub, this, std::placeholders::_1));

    // subVcuData = create_subscription<gpal_msgs::msg::VcuData>(
    //     "vcu_data",100,
    //     std::bind(&PointsFusion::vcuDataHandler, this, std::placeholders::_1));

    // subGPSRaw = create_subscription<sensor_msgs::msg::NavSatFix>(
    //     gps_topic_name, 10,
    //     std::bind(&PointsFusion::gpsRawInfoHandler, this, std::placeholders::_1));

    // finish_mapping_sub = create_subscription<std_msgs::msg::Empty>(
    //     "finish_map", 1,
    //     std::bind(&PointsFusion::finishMappingSub, this, std::placeholders::_1));

    allocateMemory();
  }

  void allocateMemory()
  {

    radarCloudRaw.reset(new pcl::PointCloud<XYZRGBSemanticsInfo>());
    pointCloudAll.reset(new pcl::PointCloud<XYZRGBSemanticsInfo>());
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    laserCloudAllFromMap.reset(new pcl::PointCloud<XYZRGBSemanticsInfo>());
  }

  // std::list<slotInfo> extractIndices(std::list<slotInfo> slots, std::vector<int> indices)
  // {
  //   for (int i = 0; i < indices.size(); ++i)
  //   {
  //     for (auto slot : slots)
  //     {
  //       if (slot.id = indices[i])
  //       {
  //         slots.remove(slot);
  //         break;
  //       }
  //     }
  //   }
  //   return slots;
  // }

  void saveSlotsSub(const std_msgs::msg::Empty::SharedPtr msg)
  {
    cout << "saving slot..." << endl;
    laserCloudAllFromMap->points.clear();
    int id = 0;
    for (int i = 0; i < allSlotsKeyFrames.size(); ++i)
    {
      list<slotInfo> curr_slot = allSlotsKeyFrames[i];
      for (list<slotInfo>::iterator itr = curr_slot.begin(); itr != curr_slot.end(); ++itr)
      {
        // 对每个车位构造点云
        assert(itr->corner_points.size() == 4);
        for (int i = 0; i < itr->corner_points.size(); ++i)
        {
          XYZRGBSemanticsInfo slot_cloud;
          slot_cloud.id = id++;
          slot_cloud.label = 1000;
          // 转化到map下
          geometry_msgs::msg::Point local_corner;
          local_corner.x = itr->corner_points[i].x;
          local_corner.y = itr->corner_points[i].y;
          geometry_msgs::msg::Point map_corner;
          transToMap(local_corner, map_corner, cloudKeyPoses6D->points[itr->keyframe_id]);
          slot_cloud.x = map_corner.x;
          slot_cloud.y = map_corner.y;
          slot_cloud.z = 1.0;
          laserCloudAllFromMap->points.push_back(slot_cloud);
          cout << itr->id << " ";
        }
      }
    }
    cout << endl;

    // 保存pcd
    pcl::io::savePCDFileBinary("all_slots.pcd", *laserCloudAllFromMap);

    // 保存轨迹
    pcl::io::savePCDFileASCII("keyframePose.pcd", *cloudKeyPoses6D);
   
    // 保存 xml
    saveXMLFile(allSlotsKeyFrames, "slot_semantic.xml");
  }

  void pathSubHandler(const nav_msgs::msg::Path::SharedPtr path)
  {
    if (isCorrectPose)
    {
      // 对关键帧pose进行更新
      cout << "update pose..." << endl;
      int start_ind = 0;
      for (int i = 0; i < cloudKeyPoses6D->points.size(); ++i)
      {
        for (int j = start_ind; j < path->poses.size(); ++j)
        {
          if (cloudKeyPoses6D->points[i].his_time == stamp2Sec(path->poses[j].header.stamp))
          {
            // cout << "correct pose..." << endl;

            cloudKeyPoses6D->points[i].x = path->poses[j].pose.position.x;
            cloudKeyPoses6D->points[i].y = path->poses[j].pose.position.y;
            cloudKeyPoses6D->points[i].z = path->poses[j].pose.position.z;
            tf2::Quaternion orientation;
            tf2::fromMsg(path->poses[j].pose.orientation, orientation);
            // 获得此时rpy
            double roll, pitch, yaw;
            tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
            cloudKeyPoses6D->points[i].roll = roll;
            cloudKeyPoses6D->points[i].pitch = pitch;
            cloudKeyPoses6D->points[i].yaw = yaw;
            start_ind = j;
            break;
          }
        }
      }

      // 更新完再判断一次
      if (cloudKeyPoses3D->points.size() > 0)
      {
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tre
        // surroundingKeyframeSearchRadius 默认设置位50
        // 搜索结果放在pointSearchInd, pointSearchSqDis  注意 后者是squared distance的意思 也即距离平方
        int num = cloudKeyPoses3D->points.size() > 20 ? 20 : cloudKeyPoses3D->points.size();
        for (int i = 0; i < num; ++i) // 检测当前关键帧前10个关键帧
        {
          // 1.搜索附近的关键帧
          pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
          std::vector<int> pointSearchInd;
          std::vector<float> pointSearchSqDis;
          std::vector<int> new_indices_pose_delete;                // 当前帧需要删除的id,根据pose
          std::map<int, std::vector<int>> his_indices_pose_delete; // 历史关键帧需要删除的id, 根据pose
          kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->points[cloudKeyPoses3D->points.size() - 1 - i], (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
          // 2. 当前帧的每一个车位和历史关键帧中车位进行id重复判断
          // 遍历搜索结果，将搜索结果的点云帧加入到surroundingKeyPoses里
          list<slotInfo> currSlots = allSlotsKeyFrames[cloudKeyPoses3D->points.size() - 1 - i];
          for (int i = 0; i < (int)pointSearchInd.size(); ++i)
          {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
            std::vector<int> indices;
            // indices.resize(100, -1);
            // his_indices_delete.insert(std::make_pair(cloudKeyPoses3D->points[id].intensity, indices));
            his_indices_pose_delete.insert(std::make_pair(cloudKeyPoses3D->points[id].intensity, indices));
          }
          for (list<slotInfo>::iterator itr = currSlots.begin(); itr != currSlots.end(); itr++)
          {
            int index = -1;
            int start = 0; // 从哪一帧开始搜索相同车位
            bool is_find_same = false;
            geometry_msgs::msg::Point local_center;
            local_center.x = itr->center_x;
            local_center.y = itr->center_y;
            geometry_msgs::msg::Point map_center;
            transToMap(local_center, map_center, cloudKeyPoses6D->points[itr->keyframe_id]);
            for (int j = 0; j < surroundingKeyPoses->points.size(); ++j)
            {
              index = surroundingKeyPoses->points[j].intensity;
              std::list<slotInfo> curr = allSlotsKeyFrames[index];
              std::vector<int> delete_vec;
              if (curr.size() == 0)
                continue;

              for (std::list<slotInfo>::iterator it = curr.begin(); it != curr.end(); it++)
              {
                if (it->id == itr->id) // 将相同的关键帧进行删除
                  continue;
                geometry_msgs::msg::Point curr_local_center;
                curr_local_center.x = it->center_x;
                curr_local_center.y = it->center_y;
                geometry_msgs::msg::Point curr_map_center;
                transToMap(curr_local_center, curr_map_center, cloudKeyPoses6D->points[it->keyframe_id]);
                double diff_dis = sqrt((map_center.x - curr_map_center.x) * (map_center.x - curr_map_center.x) +
                                       (map_center.y - curr_map_center.y) * (map_center.y - curr_map_center.y));
                // cout << "diff_dis:" << diff_dis << endl;
                if (diff_dis < 1.0)
                {
                  cout << "same slot:" << itr->id << "<--->" << it->id << endl;
                  if (itr->score >= it->score)
                  {
                    // delete_vec.push_back(it->id);
                    his_indices_pose_delete[index].push_back(it->id);
                  }
                  else
                  {
                    new_indices_pose_delete.push_back(itr->id);
                  }
                  // 找到相同id了，就不再往下搜
                  is_find_same = true;
                  break;
                }
              }
              // printListSlotId(curr);
              //  gengxin
              // allSlotsKeyFrames[index] = curr;
              if (is_find_same)
                break;
            }
          }

          // 删除重复id
          if (his_indices_pose_delete.size() > 0)
          {
            for (auto ind : his_indices_pose_delete)
            {

              for (int i = 0; i < ind.second.size(); ++i)
                eraseListSlotId(allSlotsKeyFrames[ind.first], ind.second[i]);
            }
          }

          for (int i = 0; i < new_indices_pose_delete.size(); ++i)
          {
            eraseListSlotId(currSlots, new_indices_pose_delete[i]);
          }
          allSlotsKeyFrames[cloudKeyPoses3D->points.size() - 1 - i] = currSlots;
        }
      }

      isCorrectPose = false;
    }
  }

  void correctPoseSub(const std_msgs::msg::Empty::SharedPtr msg)
  {
    isCorrectPose = true;
  }

  void printListSlotId(list<slotInfo> slot)
  {
    for (auto s : slot)
      cout << s.id << " ";
    if (slot.size() > 0)
      cout << endl;
  }

  void eraseListSlotId(list<slotInfo> &slot, int id)
  {
    for (list<slotInfo>::iterator itr = slot.begin(); itr != slot.end(); ++itr)
    {
      if (itr->id == id)
        itr = slot.erase(itr);
    }
  }

  void parkSemanticSubCB(const gpal_msgs::msg::CtlPMsgsArray::SharedPtr parkMsg)
  {

    timeRadarInfoCur = stamp2Sec(parkMsg->header.stamp);

    if (!updateInitialPose())
      return;
    // std::cout << "update initial pose succeed." << std::endl;
    //  构造当前帧车位信息
    // std::cout << "curr slot size:" << parkMsg->ctlparray.size() << std::endl;
    if (currSlots.size() != 0)
      currSlots.clear();
    for (int i = 0; i < parkMsg->ctlparray.size(); ++i)
    {
      // std::cout << parkMsg->ctlparray[i].ids << " ";
      slotInfo slot;
      slot.id = parkMsg->ctlparray[i].ids;
      slot.score = parkMsg->ctlparray[i].score;
      pcl::PointXYZ pa, pb, pc, pd;
      pa.x = parkMsg->ctlparray[i].pax;
      pa.y = parkMsg->ctlparray[i].pay;
      pb.x = parkMsg->ctlparray[i].pbx;
      pb.y = parkMsg->ctlparray[i].pby;
      pc.x = parkMsg->ctlparray[i].pcx;
      pc.y = parkMsg->ctlparray[i].pcy;
      pd.x = parkMsg->ctlparray[i].pdx;
      pd.y = parkMsg->ctlparray[i].pdy;

      slot.corner_points.push_back(pa);
      slot.corner_points.push_back(pb);
      slot.corner_points.push_back(pc);
      slot.corner_points.push_back(pd);

      slot.cc = parkMsg->ctlparray[i].cc;
      slot.time = timeRadarInfoCur;
      // slot的中心点
      slot.center_x = (pa.x + pb.x + pc.x + pd.x) / 4;
      slot.center_y = (pa.y + pb.y + pc.y + pd.y) / 4;

      if (sqrt(slot.center_x * slot.center_x + slot.center_y * slot.center_y) > 10)
        continue;

      currSlots.push_back(slot);

      // 这里可以发布当前的车位
    }
    // std::cout << std::endl;
    //  这里实时判断是否是相同id的车位；维护所有历史关键帧的ids
    //  1.在当前pose范围内的关键帧拿出来，遍历检查score,ids,
    //  1.将当前pose出一定范围内的关键帧找出来
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // std::this_thread::sleep_for(600000);
    if (isKeyFrames() && currSlots.size() > 0)
    {
      // 把当前的id的输出来
      // printListSlotId(currSlots);

      std::vector<int> new_indices_delete;                // 当前帧需要删除的id
      std::map<int, std::vector<int>> his_indices_delete; // 历史关键帧需要删除的id

      std::vector<int> new_indices_pose_delete;                // 当前帧需要删除的id,根据pose
      std::map<int, std::vector<int>> his_indices_pose_delete; // 历史关键帧需要删除的id, 根据pose

      if (cloudKeyPoses3D->points.size() > 0)
      {
        // std::cout << "judge same slot id." << std::endl;
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tre
        // surroundingKeyframeSearchRadius 默认设置位50
        // 搜索结果放在pointSearchInd, pointSearchSqDis  注意 后者是squared distance的意思 也即距离平方
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);

        // 2. 当前帧的每一个车位和历史关键帧中车位进行id重复判断
        // 遍历搜索结果，将搜索结果的点云帧加入到surroundingKeyPoses里
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
          int id = pointSearchInd[i];
          surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
          std::vector<int> indices;
          // indices.resize(100, -1);
          his_indices_delete.insert(std::make_pair(cloudKeyPoses3D->points[id].intensity, indices));
          his_indices_pose_delete.insert(std::make_pair(cloudKeyPoses3D->points[id].intensity, indices));
        }

        // printListSlotId(currSlots);
        for (list<slotInfo>::iterator itr = currSlots.begin(); itr != currSlots.end(); itr++)
        {
          // std::cout << "curr id:" << itr->id << " "
          //           << "score:" << itr->score << " his id:" << std::endl;
          int index = -1;
          int start = 0; // 从哪一帧开始搜索相同车位
          bool is_find_same = false;
          for (int j = 0; j < surroundingKeyPoses->points.size(); ++j)
          {
            index = surroundingKeyPoses->points[j].intensity;
            std::list<slotInfo> curr = allSlotsKeyFrames[index];
            std::vector<int> delete_vec;
            if (curr.size() == 0)
              continue;
            for (std::list<slotInfo>::iterator it = curr.begin(); it != curr.end(); it++)
            {
              // std::cout << it->id << " " << it->score << std::endl;
              if (itr->id == it->id)
              {
                if (itr->score >= it->score)
                {
                  // delete_vec.push_back(it->id);
                  his_indices_delete[index].push_back(it->id);
                }
                else
                {
                  new_indices_delete.push_back(itr->id);
                }
                // 找到相同id了，就不再往下搜
                is_find_same = true;
                break;
              }
            }
            // printListSlotId(curr);
            //  gengxin
            // allSlotsKeyFrames[index] = curr;
            if (is_find_same)
              break;
          }
        }

        // 删除重复id
        if (his_indices_delete.size() > 0)
        {
          for (auto ind : his_indices_delete)
          {

            for (int i = 0; i < ind.second.size(); ++i)
              eraseListSlotId(allSlotsKeyFrames[ind.first], ind.second[i]);
          }
        }

        for (int i = 0; i < new_indices_delete.size(); ++i)
        {
          eraseListSlotId(currSlots, new_indices_delete[i]);
        }
      }
      // 构造关键帧的pose
      // std::cout << "is keyFrames:" << cloudKeyPoses3D->size() << std::endl;
      PointType thisPose3D;
      thisPose3D.x = transformTobeMapped[3];
      thisPose3D.y = transformTobeMapped[4];
      thisPose3D.z = transformTobeMapped[5];
      thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index、
      // 这里终于push_back cloudKeyPoses3D
      cloudKeyPoses3D->push_back(thisPose3D);

      PointTypePose thisPose6D;
      thisPose6D.x = thisPose3D.x;
      thisPose6D.y = thisPose3D.y;
      thisPose6D.z = thisPose3D.z;
      thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
      thisPose6D.roll = transformTobeMapped[0];
      thisPose6D.pitch = transformTobeMapped[1];
      thisPose6D.yaw = transformTobeMapped[2];
      thisPose6D.time = timeRadarInfoCur;
      thisPose6D.his_time = transformTobeMapped[6];
      cloudKeyPoses6D->push_back(thisPose6D);
      keyFramePoseTimestamp.push_back(stamp2Sec(timeRadarInfoCur));
      // 关键帧点云
      // pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr thisAllKeyFrame(new pcl::PointCloud<XYZRGBSemanticsInfo>());
      // pcl::copyPointCloud(*radarCloudRaw, *thisAllKeyFrame);
      // cout << "111" << endl;
      // printListSlotId(currSlots);
      // for(auto slot : currSlots)
      //   slot.keyframe_id = thisPose3D.intensity;

      for (list<slotInfo>::iterator it = currSlots.begin(); it != currSlots.end(); ++it)
        it->keyframe_id = thisPose3D.intensity;
#if 0
      // 再根据车位中心点位置进行删除
      for (list<slotInfo>::iterator itr = currSlots.begin(); itr != currSlots.end(); itr++)
      {
        int index = -1;
        int start = 0; // 从哪一帧开始搜索相同车位
        bool is_find_same = false;
        geometry_msgs::msg::Point local_center;
        local_center.x = itr->center_x;
        local_center.y = itr->center_y;
        geometry_msgs::msg::Point map_center;
        transToMap(local_center, map_center, cloudKeyPoses6D->points[itr->keyframe_id]);
        for (int j = 0; j < surroundingKeyPoses->points.size(); ++j)
        {
          index = surroundingKeyPoses->points[j].intensity;
          std::list<slotInfo> curr = allSlotsKeyFrames[index];
          std::vector<int> delete_vec;
          if (curr.size() == 0)
            continue;
          for (std::list<slotInfo>::iterator it = curr.begin(); it != curr.end(); it++)
          {
            geometry_msgs::msg::Point curr_local_center;
            curr_local_center.x = it->center_x;
            curr_local_center.y = it->center_y;
            geometry_msgs::msg::Point curr_map_center;
            transToMap(curr_local_center, curr_map_center, cloudKeyPoses6D->points[it->keyframe_id]);
            double diff_dis = sqrt((map_center.x - curr_map_center.x) * (map_center.x - curr_map_center.x) +
                                   (map_center.y - curr_map_center.y) * (map_center.y - curr_map_center.y));
            // cout << "diff_dis:" << diff_dis << endl;
            if (diff_dis < 1.0)
            {
              cout << "same slot:" << itr->id << "<--->" << it->id << endl;
              if (itr->score >= it->score)
              {
                // delete_vec.push_back(it->id);
                his_indices_pose_delete[index].push_back(it->id);
              }
              else
              {
                new_indices_pose_delete.push_back(itr->id);
              }
              // 找到相同id了，就不再往下搜
              is_find_same = true;
              break;
            }
          }
          // printListSlotId(curr);
          //  gengxin
          // allSlotsKeyFrames[index] = curr;
          if (is_find_same)
            break;
        }
      }

      // 删除重复id
      if (his_indices_pose_delete.size() > 0)
      {
        for (auto ind : his_indices_pose_delete)
        {

          for (int i = 0; i < ind.second.size(); ++i)
            eraseListSlotId(allSlotsKeyFrames[ind.first], ind.second[i]);
        }
      }

      for (int i = 0; i < new_indices_pose_delete.size(); ++i)
      {
        eraseListSlotId(currSlots, new_indices_pose_delete[i]);
      }
#endif
      allSlotsKeyFrames.push_back(currSlots);
      // cout << "222" << endl;
      // printListSlotId(currSlots);
      publishFrames();
    }
  }

  void gpsRawInfoHandler(const sensor_msgs::msg::NavSatFix::SharedPtr gps_msg)
  {
    mtxGPS.lock();
    gpsRawQueue.push_back(*gps_msg);
  }

  void vcuDataHandler(const gpal_msgs::msg::VcuData::SharedPtr vcuDataMsg)
  {
    std::lock_guard<std::mutex> lock(vcuMtx);
    vcuDataQueue.push_back(*vcuDataMsg);
  }

  int associateKeyframeWithVcuDAata(double time)
  {
    std::lock_guard<std::mutex> lock(vcuMtx);
    if (vcuDataQueue.empty())
    {
      RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "vcuDataQueue is empty!");
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

  void poseInfoHandler(const nav_msgs::msg::Odometry::SharedPtr msgIn)
  {
    // poseMtx.lock();
    curPose = *msgIn;
    poseQueue.push_back(curPose);
    // 解锁
    //  poseMtx.unlock();
  }

  bool updateInitialPose()
  {
    // poseMtx.lock();
    while (!poseQueue.empty())
    {
      // std::cout << fixed << setprecision(8) << "timeRadarInfoCur: " << timeRadarInfoCur << ", "
      //           << "stamp2Sec(poseQueue.front().header.stamp):"
      //           << stamp2Sec(poseQueue.front().header.stamp) << " , " << stamp2Sec(poseQueue.front().header.stamp) - timeRadarInfoCur << std::endl;
      if (stamp2Sec(poseQueue.front().header.stamp) < timeRadarInfoCur)
        poseQueue.pop_front();
      else
        break;
    }

    if (poseQueue.empty())
    {
      return false;
    }
    nav_msgs::msg::Odometry curr_odom = poseQueue.front();
    tf2::Quaternion orientation;
    tf2::fromMsg(curr_odom.pose.pose.orientation, orientation);
    // 获得此时rpy
    double roll, pitch, yaw;
    tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    // 这就是我们在mapOptimization里的初始猜测，所以我们知道，这个猜测是来源于odom
    // Initial guess used in mapOptimization
    transformTobeMapped[3] = curr_odom.pose.pose.position.x;
    transformTobeMapped[4] = curr_odom.pose.pose.position.y;
    transformTobeMapped[5] = curr_odom.pose.pose.position.z;
    transformTobeMapped[0] = roll;
    transformTobeMapped[1] = pitch;
    transformTobeMapped[2] = yaw;
    // 这里再填充一个，标识用的是哪个时刻的pose
    transformTobeMapped[6] = stamp2Sec(curr_odom.header.stamp);
    // poseMtx.unlock();
    return true;
  }

  bool isKeyFrames()
  {

    if (cloudKeyPoses3D->points.empty())
      return true;

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

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
  }

  void publishFrames()
  {
    if (cloudKeyPoses3D->points.empty())
      return;
    // std::cout << "publish frames." << std::endl;
    //  publish key poses
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeRadarInfoStamp, odometryFrame);

    if (pubRecentKeyFrame->get_subscription_count() != 0)
    {
      // pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
      // PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
      // *cloudOut += *transformPointCloud(radarCloudRaw, &thisPose6D);

      // publishCloud(pubRecentKeyFrame, cloudOut, timeRadarInfoStamp, odometryFrame);
      // 发布当前车位信息
    }

    if (pubSlotInfo->get_subscription_count() != 0)
    {
      publishSlotMarker(pubSlotInfo, currSlots, timeRadarInfoStamp);
    }

    // 将所有关键帧的pose输出来
    // for (int i = 0; i < allSlotsKeyFrames.size(); ++i)
    // {
    //   cout << "keyrame" << i << "--->";
    //   printListSlotId(allSlotsKeyFrames[i]);
    // }
  }

  void publishSlotCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
                        pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
  {
  }

  void publishSlotMarker(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr thisPub,
                         std::list<slotInfo> slots, rclcpp::Time thisStamp, std::string thisFrame = "map")
  {
    //在发布之前，先把所有的marker删除掉
    visualization_msgs::msg::Marker markerD;
    markerD.header.frame_id = thisFrame;
    markerD.action = visualization_msgs::msg::Marker::DELETEALL;
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.push_back(markerD);
    vector<visualization_msgs::msg::Marker> box_vis, corner_vis, id_vis;
    fillMarkerArrayValue(slots, box_vis, corner_vis, id_vis);
    for (auto b : box_vis)
      marker_array.markers.push_back(b);
    for (auto c : corner_vis)
      marker_array.markers.push_back(c);
    for (auto id : id_vis)
      marker_array.markers.push_back(id);

    thisPub->publish(marker_array);
  }

  // void fillMarkerArrayValue(std::list<slotInfo> slot, visualization_msgs::msg::MarkerArray &marker_array)
  void fillMarkerArrayValue(std::list<slotInfo> slot, vector<visualization_msgs::msg::Marker> &box,
                            vector<visualization_msgs::msg::Marker> &corner,
                            vector<visualization_msgs::msg::Marker> &id)
  {

    for (auto s : slot)
    {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = rclcpp::Time(s.time);
      marker.ns = "parking_slot_ns_box";
      marker.id = s.id;
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;

      // 从关键帧中拿pose
      // cout << "s.keyframe_id:" << s.keyframe_id << endl;
      marker.pose.position.x = cloudKeyPoses6D->points[s.keyframe_id].x;
      marker.pose.position.y = cloudKeyPoses6D->points[s.keyframe_id].y;
      marker.pose.position.z = 1.0;

      tf2::Quaternion quat;
      geometry_msgs::msg::Quaternion ros_quat;
      quat.setRPY(0, 0, cloudKeyPoses6D->points[s.keyframe_id].yaw);
      tf2::convert(quat, ros_quat);
      marker.pose.orientation = ros_quat;
      marker.scale.x = 0.1; // 20
      // marker.scale.y = 1;  // 2
      // marker.scale.z = 1;  // 2

      marker.color.a = 1.0; // Don't forget to set the alpha!
      if (!s.cc)            // 没有占据
      {
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        // free_slots_num_++;
      }
      else
      {
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        // occ_slots_num_++;
      }

      // only if using a MESH_RESOURCE marker type:
      // marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
      // vis_pub.publish(marker);
      geometry_msgs::msg::Point first_p;
      first_p.x = s.corner_points[0].x;
      first_p.y = s.corner_points[0].y;
      geometry_msgs::msg::Point second_p;
      second_p.x = s.corner_points[1].x;
      second_p.y = s.corner_points[1].y;
      geometry_msgs::msg::Point third_p;
      third_p.x = s.corner_points[2].x;
      third_p.y = s.corner_points[2].y;
      geometry_msgs::msg::Point fourth_p;
      fourth_p.x = s.corner_points[3].x;
      fourth_p.y = s.corner_points[3].y;

      geometry_msgs::msg::Point map_pa;
      transToMap(first_p, map_pa, cloudKeyPoses6D->points[s.keyframe_id]);

      geometry_msgs::msg::Point map_pb;
      transToMap(second_p, map_pb, cloudKeyPoses6D->points[s.keyframe_id]);

      geometry_msgs::msg::Point map_pc;
      transToMap(third_p, map_pc, cloudKeyPoses6D->points[s.keyframe_id]);

      geometry_msgs::msg::Point map_pd;
      transToMap(fourth_p, map_pd, cloudKeyPoses6D->points[s.keyframe_id]);

      marker.points.push_back(first_p);
      marker.points.push_back(second_p);
      marker.points.push_back(third_p);
      marker.points.push_back(fourth_p);
      marker.points.push_back(first_p);

      // 角点
      visualization_msgs::msg::Marker corner_points_marker;
      corner_points_marker.header.frame_id = "map";
      corner_points_marker.header.stamp = rclcpp::Time(s.time);
      corner_points_marker.ns = "parking_slot_corner_points";
      corner_points_marker.id = s.id;
      corner_points_marker.type = visualization_msgs::msg::Marker::POINTS;
      corner_points_marker.action = visualization_msgs::msg::Marker::ADD;
      corner_points_marker.scale.x = 0.2;
      corner_points_marker.scale.y = 0.2;
      // corner_points_marker.scale.x = 10;
      corner_points_marker.color.g = 1.0;
      corner_points_marker.color.a = 1.0;

      corner_points_marker.pose.position.x = cloudKeyPoses6D->points[s.keyframe_id].x;
      corner_points_marker.pose.position.y = cloudKeyPoses6D->points[s.keyframe_id].y;
      corner_points_marker.pose.position.z = 1.0;

      corner_points_marker.pose.orientation = ros_quat;

      corner_points_marker.points.push_back(first_p);
      corner_points_marker.points.push_back(second_p);
      corner_points_marker.points.push_back(third_p);
      corner_points_marker.points.push_back(fourth_p);

      // id
      visualization_msgs::msg::Marker id_marker;
      id_marker.header.frame_id = "map";
      id_marker.header.stamp = rclcpp::Time(s.time);

      id_marker.ns = "parking_slot_ns_id";
      id_marker.id = s.id;

      id_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      id_marker.action = visualization_msgs::msg::Marker::ADD;
      // 这个地方应该重新计算在map下的center，因为pose变了，或者update时也要更新center
      geometry_msgs::msg::Point local_center;
      geometry_msgs::msg::Point map_center;
      local_center.x = s.center_x;
      local_center.y = s.center_y;
      transToMap(local_center, map_center, cloudKeyPoses6D->points[s.keyframe_id]);
      id_marker.pose.position.x = map_center.x;
      id_marker.pose.position.y = map_center.y;
      id_marker.pose.position.z = 1.0;
      tf2::Quaternion quat1;
      geometry_msgs::msg::Quaternion ros_quat1;
      quat1.setRPY(0, 0, cloudKeyPoses6D->points[s.keyframe_id].yaw);
      tf2::convert(quat1, ros_quat1);
      id_marker.pose.orientation = ros_quat1;

      id_marker.scale.z = 1; // 2 Only scale.z is used. scale.z specifies the height of an uppercase "A".

      id_marker.color.a = 1.0; // Don't forget to set the alpha!
      if (!s.cc)               // 没有占据
      {
        id_marker.color.r = 0.0;
        id_marker.color.g = 1.0;
        id_marker.color.b = 0.0;
      }
      else
      {
        id_marker.color.r = 1.0;
        id_marker.color.g = 0.0;
        id_marker.color.b = 0.0;
      }
      id_marker.text = std::to_string(s.id);

      // marker_array.markers.push_back(id_marker);
      // marker_array.markers.push_back(marker);
      // marker_array.markers.push_back(corner_points_marker);
      box.push_back(marker);
      corner.push_back(corner_points_marker);
      id.push_back(id_marker);
    }
  }

  pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr cloudIn, PointTypePose *transformIn)
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
      // cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  void transToMap(geometry_msgs::msg::Point in, geometry_msgs::msg::Point &out, PointTypePose curr_slot_pose)
  {
    tf2::Transform local_pose;
    local_pose.setOrigin(tf2::Vector3(curr_slot_pose.x, curr_slot_pose.y, 1.0));
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, curr_slot_pose.yaw);
    local_pose.setRotation(q);

    tf2::Transform base_pose;
    base_pose.setOrigin(tf2::Vector3(in.x, in.y, in.z));

    tf2::Transform out_trans = local_pose * base_pose;

    out.x = out_trans.getOrigin().x();
    out.y = out_trans.getOrigin().y();

    // 这里取个巧，用z来存yaw
    double roll, pitch, yaw;
    tf2::Matrix3x3(out_trans.getRotation()).getRPY(roll, pitch, yaw);
    out.z = yaw;
    return;
  }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  void visualizeGlobalMapThread()
  {
    // 5s执行一次
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }
  }

  void publishGlobalMap()
  {
    // publish all keyframe
    // 在发布之前，先把所有的marker删除掉
    visualization_msgs::msg::Marker markerD;
    markerD.header.frame_id = "map";
    markerD.action = visualization_msgs::msg::Marker::DELETEALL;
    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.push_back(markerD);

    for (int i = 0; i < allSlotsKeyFrames.size(); ++i)
    {
      // cout << "keyframe:" << i << ":";
      // printListSlotId(allSlotsKeyFrames[i]);
      vector<visualization_msgs::msg::Marker> box_vis, corner_vis, id_vis;
      fillMarkerArrayValue(allSlotsKeyFrames[i], box_vis, corner_vis, id_vis);
      for (auto b : box_vis)
        marker_array.markers.push_back(b);
      for (auto c : corner_vis)
        marker_array.markers.push_back(c);
      for (auto id : id_vis)
        marker_array.markers.push_back(id);
    }
    pubAllSlotInfo->publish(marker_array);
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
      // is_save_map_succeed = saveMap(map_save_dir, 0.0);
      // rclcpp::sleep_for(std::chrono::milliseconds(200));
    }
    if (!is_save_map_succeed)
    {
      RCLCPP_ERROR(get_logger(), "in 5s, save map failure, please check your map_dir config or call save map by manual!");
    }
    else
    {
      RCLCPP_INFO(get_logger(), "save map succeed!");
      // 向定位节点发布建图结束的命令 连续发布10帧
      // for (int i = 0; i < 10; ++i)
      // {
      //   std_msgs::msg::Empty msg;
      //   notisfy_local_pub->publish(msg);
      //   rclcpp::sleep_for(std::chrono::milliseconds(100));
      // }
    }
  }

  void saveXMLFile(vector<list<slotInfo>> all_slots, string file_n)
  {
    XMLDocument *xmlDoc = new XMLDocument();
    XMLNode *pRoot = xmlDoc->NewElement("gpalsemanticmap");
    xmlDoc->InsertFirstChild(pRoot);
    for (int i = 0; i < all_slots.size(); ++i)
    {
      list<slotInfo> curr_slot = all_slots[i];
      for (list<slotInfo>::iterator itr = curr_slot.begin(); itr != curr_slot.end(); ++itr)
      {

        XMLElement *pElement = xmlDoc->NewElement("element");
        XMLElement *category = xmlDoc->NewElement("category");
        category->SetText("parkingslot");
        pElement->InsertEndChild(category);

        // XMLElement *type = xmlDoc->NewElement("type");
        // type->SetText("水平");
        // pElement->InsertEndChild(type);

        XMLElement *id = xmlDoc->NewElement("id");
        id->SetText(itr->id);
        pElement->InsertEndChild(id);
        XMLElement *pose = xmlDoc->NewElement("pose");
        for (int j = 0; j < 4; ++j)
        {
          geometry_msgs::msg::Point local_corner;
          local_corner.x = itr->corner_points[j].x;
          local_corner.y = itr->corner_points[j].y;
          geometry_msgs::msg::Point map_corner;
          transToMap(local_corner, map_corner, cloudKeyPoses6D->points[itr->keyframe_id]);

         
          XMLElement *pa = xmlDoc->NewElement("corner");
          XMLElement *pax = xmlDoc->NewElement("x");
          pax->SetText(map_corner.x);
          pa->InsertEndChild(pax);
          XMLElement *pay = xmlDoc->NewElement("y");
          pay->SetText(map_corner.y);
          pa->InsertEndChild(pay);
          pose->InsertEndChild(pa);
        }
        pElement->InsertEndChild(pose);

        XMLElement *score = xmlDoc->NewElement("score");
        score->SetText(itr->score);
        pElement->InsertEndChild(score);

        XMLElement *occ = xmlDoc->NewElement("occ");
        occ->SetText(itr->cc);
        pElement->InsertEndChild(occ);

        pRoot->InsertEndChild(pElement);
      }
    }

    xmlDoc->SaveFile(file_n.c_str());
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  rclcpp::executors::MultiThreadedExecutor exec;

  auto IP = std::make_shared<PointsFusion>(options);
  exec.add_node(IP);

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> PointFusion Started.\033[0m");
  std::thread visualizeMapThread(&PointsFusion::visualizeGlobalMapThread, IP);

  exec.spin();

  visualizeMapThread.join();
  rclcpp::shutdown();
  return 0;
}