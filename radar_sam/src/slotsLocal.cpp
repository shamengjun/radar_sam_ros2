#include "utility.h"
#include <Eigen/Dense> //四元数转旋转向量
#include "boxFilter/boxFilter.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <visualization_msgs/msg/marker_array.hpp>

// cost function : \sum|(Rq_i + t - p_i)|^2
struct PPICP
{
    PPICP(const Eigen::Vector2d q, const Eigen::Vector2d p)
        : q(q), p(p) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        T p_q[2];
        T R[2][2];
        R[0][0] = cos(pose[2]);
        R[0][1] = -sin(pose[2]);
        R[1][0] = sin(pose[2]);
        R[1][1] = cos(pose[2]);
        p_q[0] = T(q(0));
        p_q[1] = T(q(1));
        T p_pro[2];
        p_pro[0] = R[0][0] * T(q(0)) + R[0][1] * T(q(1));
        p_pro[1] = R[1][0] * T(q(0)) + R[1][1] * T(q(1));

        p_pro[0] += pose[0];
        p_pro[1] += pose[1];

        T diff[2];
        diff[0] = p_pro[0] - T(p(0));
        diff[1] = p_pro[1] - T(p(1));

        residuals[0] = diff[0] * diff[0] + diff[1] * diff[1];
        return true;
    }

    static ceres::CostFunction *create(const Eigen::Vector2d q, const Eigen::Vector2d p)
    {
        return (new ceres::AutoDiffCostFunction<PPICP, 1, 3>(new PPICP(q, p)));
    }
    Eigen::Vector2d q;
    Eigen::Vector2d p;
};

class slotsLocalization : public ParamServer
{
private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subSlotPoints;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subInitialPose;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubLocalPose;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLocalMap;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubGlobalMap;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCurrSlotsPoints;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubCurrSlotInfo;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubMatchSlotInfo;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSourcePoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubTargetPoints;
    

    pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr globalSlotsMap;
    pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr localSlotsMap;
    std::string slotMapDir = "/media/zhaoyz/code_ws/radar_slam_ws2/all_slots.pcd";
    bool loadedSlotMap = false;
    Eigen::Matrix4d priorPose = Eigen::Matrix4d::Identity();
    std::deque<nav_msgs::msg::Odometry> poseQueue;
    std::mutex poseMutex;
    double currSlotsTime;
    std::shared_ptr<BoxFilter> boxFilter;

    std::shared_ptr<std::thread> visualizeGlobalMapThread;

public:
    slotsLocalization(const rclcpp::NodeOptions &options) : ParamServer("radar_sam_slotsLocalization", options)
    {
        subSlotPoints = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/park_points", qos,
            std::bind(&slotsLocalization::slotsCloudInfoHandler, this, std::placeholders::_1));
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "opt_odom", qos,
            std::bind(&slotsLocalization::poseInfoHandler, this, std::placeholders::_1));
        subInitialPose = create_subscription<nav_msgs::msg::Odometry>(
            "initial_pose", qos,
            std::bind(&slotsLocalization::initialPoseHandler, this, std::placeholders::_1));

        pubLocalPose = create_publisher<nav_msgs::msg::Path>("slot_local_path", 1);

        pubLocalMap = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", 1);

        pubGlobalMap = create_publisher<sensor_msgs::msg::PointCloud2>("global_map", 1);

        pubSourcePoints = create_publisher<sensor_msgs::msg::PointCloud2>("source_points", 1);

        pubTargetPoints = create_publisher<sensor_msgs::msg::PointCloud2>("target_points", 1);

        pubCurrSlotsPoints = create_publisher<sensor_msgs::msg::PointCloud2>("curr_slots", 1);

        pubCurrSlotInfo = create_publisher<visualization_msgs::msg::MarkerArray>("/curr_slots_marker", 1);
        pubMatchSlotInfo = create_publisher<visualization_msgs::msg::MarkerArray>("/match_slots_marker", 1);

        globalSlotsMap.reset(new pcl::PointCloud<XYZRGBSemanticsInfo>());
        localSlotsMap.reset(new pcl::PointCloud<XYZRGBSemanticsInfo>());

        if (!loadGlobalSlotMap(slotMapDir))
        {
            std::cout << "load global map from:" << slotMapDir << " failed!" << std::endl;
        }
        else
        {
            for (int i = 0; i < globalSlotsMap->points.size(); ++i)
            {
                cout << globalSlotsMap->points[i].id << " ";
            }
            cout << endl;

            loadedSlotMap = true;
        }
          
        // 定义获取local map的范围，min_x, max_x, min_y, max_y, min_z, max_z
        std::vector<double> box_size{-20.0, 20.0, -10.0, 10.0, -10.0, 10.0};
        boxFilter = std::make_shared<BoxFilter>(box_size);

        // //初始时，resetLocalMap 为0 0 0
        // if (resetLocalMap(0.0, 0.0, 0.0))
        // {
        //    std::cout << "initial reset local map succeed." << std::endl;
        // }
        // else
        // {
        //    std::cerr << "initial reset local map failed." << std::endl;
        // }
        visualizeGlobalMapThread = std::make_shared<std::thread>(&slotsLocalization::visualizeGlobalMap, this);
    }

    bool loadGlobalSlotMap(const std::string &slot_map_file)
    {
        int ret = pcl::io::loadPCDFile(slot_map_file, *globalSlotsMap);
        std::cout << "ret:" << ret << std::endl;
        if (ret != -1)
        {
            std::cout << "load slots pcd points:" << globalSlotsMap->points.size() << std::endl;
            return true;
        }
        return false;
    }

    void slotsCloudInfoHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msgIn)
    {
        static int slot_msg_cnt = 0;
        // cout << "slot msg cnt:" << slot_msg_cnt++ << endl;
        currSlotsTime = stamp2Sec(msgIn->header.stamp);
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr currSlotPoints(new pcl::PointCloud<XYZRGBSemanticsInfo>());
        pcl::fromROSMsg(*msgIn, *currSlotPoints);
        if (currSlotPoints->empty())
        {
            std::cerr << "slot points is empty!" << std::endl;
            return;
        }
        std::cout << "acquire prior pose." << std::endl;
        // 1. 获取先验的pose
        if (!acquirePriorPose())
        {
            std::cout << "acquire prior pose failure!" << std::endl;
            return;
        }

        // std::cout << "aquire prior pose:\n"
        //           << priorPose << std::endl;
        // 2.在priorPose附近构造匹配的车位点
        // pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr nearSlotsPoints;
        // extractNearSlotsPoints(priorPose, nearSlotsPoints);
        if (localSlotsMap->points.size() < 0)
        {
            std::cout << "near slots points is too little." << std::endl;
            return;
        }
        else
        {
            sensor_msgs::msg::PointCloud2 local_map;
            pcl::toROSMsg(*localSlotsMap, local_map);
            local_map.header.frame_id = "map";
            local_map.header.stamp = msgIn->header.stamp;
            pubLocalMap->publish(local_map);
        }
        // 3.将当前车位点变换到先验位置下
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr currGlobalPoints(new pcl::PointCloud<XYZRGBSemanticsInfo>());
        pcl::transformPointCloud(*currSlotPoints, *currGlobalPoints, priorPose);

        // 发布出来
        if (currGlobalPoints->points.size() > 0)
        {
            sensor_msgs::msg::PointCloud2 curr_slots_points;
            pcl::toROSMsg(*currGlobalPoints, curr_slots_points);
            curr_slots_points.header.frame_id = "map";
            curr_slots_points.header.stamp = msgIn->header.stamp;
            pubCurrSlotsPoints->publish(curr_slots_points);
        }

        // std::cout << "transformed to prior pose." << std::endl;

        // 4. 调用匹配
        double match_pose[3] = {0.0, 0.0, 0.0};
        slotsMatch(localSlotsMap, currGlobalPoints, match_pose);
        cout << "match pose:" << match_pose[0] << " " << match_pose[1] << " " << match_pose[2] * 180 / M_PI << endl;

        // 5. 更新先验pose
        Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
        poseToEigen(match_pose, delta);
        cout << "delta pose:\n"
             << delta << endl;
        Eigen::Matrix4d update_pose = Eigen::Matrix4d::Identity();
        update_pose = priorPose * delta;
        // cout << "update pose:\n"
        //      << update_pose << endl;

        // 6. 发布更新后的pose
        static nav_msgs::msg::Path local_path_msg;
        local_path_msg.header.stamp = msgIn->header.stamp;
        local_path_msg.header.frame_id = "map";
        auto p = geometry_msgs::msg::PoseStamped();
        p.header = local_path_msg.header;

        geometry_msgs::msg::Pose pose;
        EigenToGoemetryPose(update_pose, pose);
        p.pose = pose;

        local_path_msg.poses.push_back(p);
        pubLocalPose->publish(local_path_msg);

        // 7. 判断是否需要更新局部地图
        if (updateLocalMap(update_pose, 20.0))
        {
            std::cout << "update local map, origin:" << update_pose(0, 3) << " " << update_pose(1, 3) << " "
                      << update_pose(2, 3) << std::endl;
            if (resetLocalMap(update_pose(0, 3), update_pose(1, 3), update_pose(2, 3)))
            {
                std::cout << "update local map succeed." << std::endl;
            }
            else
            {
                std::cerr << "update local map failure." << std::endl;
            }
        }
    }

    void poseInfoHandler(const nav_msgs::msg::Odometry::SharedPtr pose)
    {
        std::lock_guard<std::mutex> lock(poseMutex);
        poseQueue.push_back(*pose);
    }

    bool acquirePriorPose()
    {
        while (!poseQueue.empty())
        {
            if (stamp2Sec(poseQueue.front().header.stamp) < currSlotsTime)
                poseQueue.pop_front();
            else
                break;
        }
        std::cout << "pose queue is not empty!" << std::endl;
        if (!poseQueue.empty())
        {
            nav_msgs::msg::Odometry odom_msg = poseQueue.front();
            // 转化成Eigen::Matrix4d的形式
            odometryToMatrix(odom_msg, priorPose);

            // 如果是第一帧，则作为第一个点
            // 初始时，resetLocalMap 为0 0 0
            static bool first_frame = false;
            if (!first_frame)
            {
                if (resetLocalMap(priorPose(0, 3), priorPose(1, 3), priorPose(2, 3)))
                {
                    std::cout << "initial reset local map succeed." << std::endl;
                    first_frame = true;
                }
                else
                {
                    std::cerr << "initial reset local map failed." << std::endl;
                }
            }

            return true;
        }

        return false;
    }

    void odometryToMatrix(const nav_msgs::msg::Odometry odom, Eigen::Matrix4d &matrix)
    {
        // 由transform获取位移矩阵 Eigen::Translation3f  (3 * 3)
        // Eigen::Translation3d tl_btol(odom.pose.position.x, odom.pose.position.y, odom.pose.positiion.z);
        // // 旋转通过以下形式从TF获得:
        // double roll, pitch, yaw;
        // tf::Matrix3x3(odom.pose.orientation).getEulerYPR(yaw, pitch, roll);
        // Eigen::AngleAxisf rot_x_btol(roll, Eigen::Vector3f::UnitX()); // 轴角表达形式
        // Eigen::AngleAxisf rot_y_btol(pitch, Eigen::Vector3f::UnitY());
        // Eigen::AngleAxisf rot_z_btol(yaw, Eigen::Vector3f::UnitZ());

        // matrix = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
        tf2::Quaternion tf_quat(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w);
        // tf2::Quaternion tf_quat(0.0, 0.0, 0.0, 1.0);
        //  tf2::quaternionMsgToTF(odom.pose.pose.orientation, tf_quat);
        tf2::Matrix3x3 tf_matrix;
        tf_matrix.setRotation(tf_quat);
        for (int i = 0; i < 3; ++i) // row
        {
            for (int j = 0; j < 3; ++j) // colomn
            {
                matrix(i, j) = tf_matrix[i][j];
            }
        }
        matrix(0, 3) = odom.pose.pose.position.x;
        matrix(1, 3) = odom.pose.pose.position.y;
        matrix(2, 3) = odom.pose.pose.position.z;
    }

    void extractNearSlotsPoints(const Eigen::Matrix4d prior_pose, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr near_slots)
    {
        // 根据当前pose切出来一个box,用到CropBox Filter, 这个函数暂且不用
    }

    void initialPoseHandler(const nav_msgs::msg::Odometry::SharedPtr initial_pose)
    {
        std::cout << "initial pose received!" << std::endl;
        const auto &p = initial_pose->pose.pose.position;
        const auto &q = initial_pose->pose.pose.orientation;
        // 加载local map
        if (resetLocalMap(p.x, p.y, p.z))
        {
            std::cout << "reset local map succeed." << std::endl;
        }
        else
        {
            std::cout << "reset local map failure." << std::endl;
        }
    }

    bool resetLocalMap(const double &x, const double &y, const double &z)
    {
        std::cout << "reset local map!" << std::endl;
        // use roi filtering for local map segmentation
        std::vector<double> origin = {x, y, z};
        boxFilter->setOrigin(origin);
        if (globalSlotsMap->size() == 0)
        {
            std::cerr << "global map empty...." << std::endl;
            return false;
        }

        boxFilter->filter(globalSlotsMap, localSlotsMap);
        std::cout << "local slots map size:" << localSlotsMap->size() << std::endl;
        return true;
    }

    void slotsMatch(pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr ref, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr per,
                    double *match_result)
    {
        if (ref->points.size() < 10)
        {
            std::cout << "ref points is less than 10, wrong!" << std::endl;
            return;
        }
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr source(new pcl::PointCloud<XYZRGBSemanticsInfo>());
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr target(new pcl::PointCloud<XYZRGBSemanticsInfo>());
#if 1
        ceres::Problem problem;
        pcl::KdTreeFLANN<XYZRGBSemanticsInfo> kdtree;
        kdtree.setInputCloud(ref->makeShared());
        int K = 1;
        std::vector<int> index(K);
        std::vector<float> dis(K);
        std::cout << "per size:" << per->points.size() << " ref size:" << ref->points.size() << std::endl;
        // std::cout << "add constraint:";
        //@todo 1. 用icp/ndt匹配看下效果
        //  2. 在如下对应点关联时添加规则，比如检测相同id点和之前关联点id是否符合。
        //  1.将当前点云按照id进行归类
        //  2.对每个id的点进行最近邻搜索
        //  3.判断当前id和的4个角点搜索到的近邻点是否也是同一个id;
        //  4.如果是，则将4个角点加入优化问题中，否则表示关联不成功。
        std::map<int, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr> id_class;
        for (int i = 0; i < per->points.size(); i++)
        {
            XYZRGBSemanticsInfo curr_p = per->points[i];
            int slot_id = per->points[i].id;
            std::map<int, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr>::iterator itr = id_class.find(slot_id);
            if (itr != id_class.end()) // 找到了
            {
                itr->second->points.push_back(curr_p);
            }
            else
            {
                pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr slots_points(new pcl::PointCloud<XYZRGBSemanticsInfo>());
                slots_points->points.push_back(curr_p);
                id_class.insert(std::pair<int, pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr>(slot_id, slots_points));
            }
        }

        //

        // 打印一下
        // for (auto itr = id_class.begin(); itr != id_class.end(); ++itr)
        // {
        //     cout << "slot id:" << itr->first << " slot corner points size:" << itr->second->points.size() << endl;
        // }

        for (auto iter = id_class.begin(); iter != id_class.end(); ++iter)
        {
            std::vector<int> ref_index;
            //cout << "per id:" << iter->first << endl;
            for (int i = 0; i < iter->second->points.size(); i++)
            {
                static int global_id = -1;
                if (kdtree.nearestKSearch(iter->second->points[i], K, index, dis) == K)
                {
                    //cout << "nearest dis:" << dis[0] << endl;
                    if (dis[0] > 1.0)
                        continue;
                    // 在地图中的id
                     if (i == 0) // 第一个点
                     {
                         global_id = ref->points[index[0]].id;
                         ref_index.push_back(index[0]);
                         continue;
                     }
                     cout << "global id:" << global_id << " ref id:" << ref->points[index[0]].id << endl;
                     if (global_id != ref->points[index[0]].id)
                         break;
                     else
                         ref_index.push_back(index[0]);
                    //global_id = ref->points[index[0]].id;
                    //cout << "global id:" << global_id << endl;
                    // 在ref中找相同global id的点
                    // for (int i = 0; i < ref->points.size(); ++i) // ref中有相同的3帧的数据，具体问题还需要排查！
                    // {
                    //     //cout << ref->points[i].id << " ";
                    //     if (ref->points[i].id == global_id)
                    //     {
                    //         ref_index.push_back(i);
                    //     }
                    // }
                    //cout << "\nfind ref same size:" << ref_index.size() << endl;
                }
                else
                {
                    cerr << "nearest search failed!" << endl;
                }
            }

            cout << "ref index size:" << ref_index.size() << " ref points size:" <<  ref->points.size() << endl;
            for (int i = 0; i < ref_index.size(); ++i)
            {
                cout << ref_index[i] << " ";
            }
            cout << endl;

            // 为了查看关联的车位，需要开发关联车位显示部分
            if (ref_index.size() == 4 /*&& std::unique(ref_index.begin(), ref_index.end()) != ref_index.end()*/)
            {
                cout << "find one slot pair!" << endl;
               // publishSlotMarker(pubCurrSlotInfo, *iter->second);
                pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr ref_points(new pcl::PointCloud<XYZRGBSemanticsInfo>());
                for (int i = 0; i < 4; ++i)
                {
                  //ref_points->points.push_back(ref->points[ref_index[i]]);
                  target->points.push_back(ref->points[ref_index[i]]);
                }
                //publishSlotMarker(pubMatchSlotInfo, *ref_points, false);
                for (int i = 0; i < iter->second->points.size(); ++i)
                {
                    source->points.push_back(iter->second->points[i]);
                }

                for (int i = 0; i < 4; ++i)
                {
                    // add constraints
                    Eigen::Vector2d q = Eigen::Vector2d(iter->second->points[i].x, iter->second->points[i].y);
                    Eigen::Vector2d p = PCL2Eigen(ref->points[ref_index[i]]);
                    // // 根据距离判断，如果太大则认为关联不成功
                    // if (distance(q, p) > 0.1)
                    // {
                    //     continue;
                    // }
                    // std::cout << i << " ";
                    ceres::CostFunction *cost_function = PPICP::create(q, p);
                    problem.AddResidualBlock(cost_function,
                                             nullptr,
                                             match_result);
                }
                //在这个地方将关联上的车位显示出来；
               // break;
            }
        }
        
        //发布source points
        sensor_msgs::msg::PointCloud2 source_points;
        pcl::toROSMsg(*source, source_points);
        source_points.header.frame_id = "map";
        source_points.header.stamp = rclcpp::Time(currSlotsTime);
        pubSourcePoints->publish(source_points);

        //发布target points
        sensor_msgs::msg::PointCloud2 target_points;
        pcl::toROSMsg(*target, target_points);
        target_points.header.frame_id = "map";
        target_points.header.stamp = rclcpp::Time(currSlotsTime);
        pubTargetPoints->publish(target_points);

        std::cout << endl;
        //ppicp
        // ceres::Solver::Options options;
        // options.linear_solver_type = ceres::DENSE_SCHUR;
        // options.minimizer_progress_to_stdout = false;
        // options.max_num_iterations = 100;

        // ceres::Solver::Summary summary;
        // ceres::Solve(options, &problem, &summary);
        // std::cout << summary.BriefReport() << "\n";
#endif
#if 0
        pcl::GeneralizedIterativeClosestPoint<XYZRGBSemanticsInfo, XYZRGBSemanticsInfo>::Ptr gicp(new pcl::GeneralizedIterativeClosestPoint<XYZRGBSemanticsInfo, XYZRGBSemanticsInfo>());
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr aligned(new pcl::PointCloud<XYZRGBSemanticsInfo>());

        gicp->setInputTarget(target);
        gicp->setInputSource(source);
        gicp->setCorrespondenceRandomness(4);
        gicp->setMaximumOptimizerIterations(100);
        gicp->setRotationEpsilon(1e-2);
        gicp->align(*aligned);
        Eigen::Matrix4d align_result = Eigen::Matrix4d::Identity();
        align_result = gicp->getFinalTransformation().cast<double>();
        Eigen::Matrix3d rotation = align_result.block<3, 3>(0, 0);
        match_result[0] = align_result(0, 3);
        match_result[1] = align_result(1, 3);
        // 旋转矩阵转换为欧拉角
        Eigen::Vector3d eulerAngle = rotation.eulerAngles(2, 1, 0); // rpy
        match_result[2] = eulerAngle(2);
#endif

#if 0
        pcl::NormalDistributionsTransform<XYZRGBSemanticsInfo, XYZRGBSemanticsInfo> ndt;
        ndt.setTransformationEpsilon(0.001); // 0.01
        ndt.setStepSize(0.001);
        ndt.setResolution(0.01); //1.0
        // ndt.setRANSACIterations(5);
        // ndt.setRANSACOutlierRejectionThreshold(0.05); // default 0.05m
        ndt.setMaximumIterations(100);
        ndt.setInputSource(target);
        ndt.setInputTarget(source);
        Eigen::Matrix4d align_result = Eigen::Matrix4d::Identity();
        if (ndt.hasConverged())
        {
           cout << "ndt converged. score:" << ndt.getFitnessScore() << endl;
           align_result = ndt.getFinalTransformation().cast<double>(); 
        }
        else
        {
           cout << "not converged"  << endl;
        }
          
        Eigen::Matrix3d rotation = align_result.block<3, 3>(0, 0);
        match_result[0] = align_result(0, 3);
        match_result[1] = align_result(1, 3);
        // 旋转矩阵转换为欧拉角
        Eigen::Vector3d eulerAngle = rotation.eulerAngles(2, 1, 0); // rpy
        match_result[2] = eulerAngle(2);
#endif
#if 1
        pcl::IterativeClosestPoint<XYZRGBSemanticsInfo, XYZRGBSemanticsInfo>::Ptr icp(new pcl::IterativeClosestPoint<XYZRGBSemanticsInfo, XYZRGBSemanticsInfo>());
        pcl::PointCloud<XYZRGBSemanticsInfo>::Ptr aligned(new pcl::PointCloud<XYZRGBSemanticsInfo>());
        
    
        icp->setInputTarget(target);
        icp->setInputSource(source);
        icp->align(*aligned);
        Eigen::Matrix4d align_result = Eigen::Matrix4d::Identity();
        align_result = icp->getFinalTransformation().cast<double>();
        Eigen::Matrix3d rotation = align_result.block<3, 3>(0, 0);
        match_result[0] = align_result(0, 3);
        match_result[1] = align_result(1, 3);
        // 旋转矩阵转换为欧拉角
        Eigen::Vector3d eulerAngle = rotation.eulerAngles(2, 1, 0); // rpy
        match_result[2] = eulerAngle(2);
#endif
        // Eigen::Matrix<double, 6, 6> HX;
        // registration->getHessian(HX);
    }

    Eigen::Vector2d PCL2Eigen(const XYZRGBSemanticsInfo &p)
    {
        Eigen::Vector2d pp;
        pp(0) = p.x;
        pp(1) = p.y;
        return pp;
    }

    double distance(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2)
    {
        return sqrt((p1(0) - p2(0)) * (p1(0) - p2(0)) + (p1(1) - p2(1)) * (p1(1) - p2(1)));
    }

    void poseToEigen(const double *pose, Eigen::Matrix4d &mat)
    {
        Eigen::Matrix3d rot_mat;
        rot_mat << 1, 0, 0,
            0, cos(pose[2]), -sin(pose[2]),
            0, sin(pose[2]), cos(pose[2]);

        Eigen::Vector3d trans_vec{pose[0], pose[1], 1.0};
        mat.block<3, 3>(0, 0) = rot_mat;
        mat.block<3, 1>(0, 3) = trans_vec;
    }

    void EigenToGoemetryPose(const Eigen::Matrix4d mat, geometry_msgs::msg::Pose &pose)
    {
        Eigen::Matrix3d rotationMatrix = mat.block<3, 3>(0, 0);
        // std::cout << "rotation:\n"
        //           << rotationMatrix << std::endl;
        Eigen::Quaterniond quaternion(rotationMatrix);
        // std::cout << "x y z:" << mat(0, 3) << " " << mat(1, 3) << " " << mat(2, 3) << std::endl;
        // std::cout << "qx qy qz qw:" << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << " " << quaternion.w() << std::endl;
        pose.position.x = mat(0, 3);
        pose.position.y = mat(1, 3);
        pose.position.z = mat(2, 3);

        pose.orientation.x = quaternion.x();
        pose.orientation.y = quaternion.y();
        pose.orientation.z = quaternion.z();
        pose.orientation.w = quaternion.w();
    }

    bool updateLocalMap(const Eigen::Matrix4d &curr_pose, const double update_dis_thres)
    {
        std::vector<double> edge = boxFilter->getEdge();
        cout << edge.at(0) << " " << edge.at(1) << " " << edge.at(2) << " "
             << edge.at(3) << " " << edge.at(4) << " " << edge.at(5) << endl;
        for (int i = 0; i < 3; ++i)
        {
            if (fabs(curr_pose(i, 3) - edge.at(2 * i)) > update_dis_thres &&
                fabs(curr_pose(i, 3) - edge.at(2 * i + 1)) > update_dis_thres)
            {
                continue;
            }
            return true;
        }
        return false;
    }

    void visualizeGlobalMap()
    {
        rclcpp::Rate rate(0.1);
        while (rclcpp::ok())
        {
            sensor_msgs::msg::PointCloud2 global_map;
            pcl::toROSMsg(*globalSlotsMap, global_map);
            global_map.header.frame_id = "map";
            // global_map.header.stamp = rclcpp::Time(currSlotsTime);
            global_map.header.stamp = this->get_clock()->now();
            pubGlobalMap->publish(global_map);
            rate.sleep();
        }
    }

    void publishSlotMarker(rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr thisPub,
                           pcl::PointCloud<XYZRGBSemanticsInfo> slots, bool source = true, std::string thisFrame = "map")
    {
        // 在发布之前，先把所有的marker删除掉
        visualization_msgs::msg::Marker markerD;
        markerD.header.frame_id = thisFrame;
        markerD.action = visualization_msgs::msg::Marker::DELETEALL;
        visualization_msgs::msg::MarkerArray marker_array;
        marker_array.markers.push_back(markerD);
        vector<visualization_msgs::msg::Marker> box_vis, corner_vis;
        fillMarkerArrayValue(slots, box_vis, corner_vis, source);
        for (auto b : box_vis)
            marker_array.markers.push_back(b);
        for (auto c : corner_vis)
            marker_array.markers.push_back(c);
        // for (auto id : id_vis)
        //     marker_array.markers.push_back(id);

        thisPub->publish(marker_array);
    }

    void fillMarkerArrayValue(pcl::PointCloud<XYZRGBSemanticsInfo> slots, vector<visualization_msgs::msg::Marker> &box,
                              vector<visualization_msgs::msg::Marker> &corner,
                              bool source)
    {

        // for (int i = 0; i < slots->points.size(); ++i)
        // {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = rclcpp::Time(slots.points[0].t);
            marker.ns = "parking_slot_ns_box";
            marker.id = slots.points[0].id;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;

            // 从关键帧中拿pose
            // cout << "s.keyframe_id:" << s.keyframe_id << endl;
            marker.pose.position.x = 0;
            marker.pose.position.y = 0;
            marker.pose.position.z = 0;

            tf2::Quaternion quat;
            geometry_msgs::msg::Quaternion ros_quat;
            quat.setRPY(0, 0, 0);
            tf2::convert(quat, ros_quat);
            marker.pose.orientation = ros_quat;
            marker.scale.x = 0.1; // 20
            // marker.scale.y = 1;  // 2
            // marker.scale.z = 1;  // 2

            marker.color.a = 1.0; // Don't forget to set the alpha!
            if (!source)            // 没有占据
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
            first_p.x = slots.points[0].x;
            first_p.y = slots.points[0].y;
            geometry_msgs::msg::Point second_p;
            second_p.x = slots.points[1].x;
            second_p.y = slots.points[1].y;
            geometry_msgs::msg::Point third_p;
            third_p.x = slots.points[1].x;
            third_p.y = slots.points[1].y;
            geometry_msgs::msg::Point fourth_p;
            fourth_p.x = slots.points[1].x;
            fourth_p.y = slots.points[1].y;

            // geometry_msgs::msg::Point map_pa;
            // transToMap(first_p, map_pa, cloudKeyPoses6D->points[s.keyframe_id]);

            // geometry_msgs::msg::Point map_pb;
            // transToMap(second_p, map_pb, cloudKeyPoses6D->points[s.keyframe_id]);

            // geometry_msgs::msg::Point map_pc;
            // transToMap(third_p, map_pc, cloudKeyPoses6D->points[s.keyframe_id]);

            // geometry_msgs::msg::Point map_pd;
            // transToMap(fourth_p, map_pd, cloudKeyPoses6D->points[s.keyframe_id]);

            marker.points.push_back(first_p);
            marker.points.push_back(second_p);
            marker.points.push_back(third_p);
            marker.points.push_back(fourth_p);
            marker.points.push_back(first_p);

            // 角点
            visualization_msgs::msg::Marker corner_points_marker;
            corner_points_marker.header.frame_id = "map";
            corner_points_marker.header.stamp = rclcpp::Time(slots.points[0].t);
            corner_points_marker.ns = "parking_slot_corner_points";
            corner_points_marker.id = slots.points[0].id;
            corner_points_marker.type = visualization_msgs::msg::Marker::POINTS;
            corner_points_marker.action = visualization_msgs::msg::Marker::ADD;
            corner_points_marker.scale.x = 0.2;
            corner_points_marker.scale.y = 0.2;
            // corner_points_marker.scale.x = 10;
            corner_points_marker.color.g = 1.0;
            corner_points_marker.color.a = 1.0;

            corner_points_marker.pose.position.x = 0.0;
            corner_points_marker.pose.position.y = 0.0;
            corner_points_marker.pose.position.z = 0.0;

            corner_points_marker.pose.orientation = ros_quat;

            corner_points_marker.points.push_back(first_p);
            corner_points_marker.points.push_back(second_p);
            corner_points_marker.points.push_back(third_p);
            corner_points_marker.points.push_back(fourth_p);
    }
};

int main(int argc, char **argv)
{

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto vcuGpsOpt = std::make_shared<slotsLocalization>(options);

    exec.add_node(vcuGpsOpt);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m---->slot localization Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();

    return 0;
}
