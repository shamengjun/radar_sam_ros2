cmake_minimum_required(VERSION 3.8)
project(radar_sam)
# set(CMAKE_CXX_STANDARD 14)
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_FLAGS "-std=c++11") //default

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")
find_package(ament_cmake REQUIRED)
find_package(gpal_msgs REQUIRED)
find_package(radar_msgs REQUIRED)
find_package(gpal_vision_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
# set(OpenCV_STATIC ON)
# find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(Boost REQUIRED)
find_package(GeographicLib REQUIRED)

#打印ndt_omp地址
message(STATUS "ndt_omp_LIBRARIES: ${ndt_omp_LIBRARIES}")
message(STATUS "Ceres libraries: ${CERES_LIBRARIES}")
# include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
  ${GeographicLib_INCLUDE_DIRS}
  ${gpal_msgs_INCLUDE_DIRS}
  ${radar_msgs_INCLUDE_DIRS}
  ${gpal_vision_msgs_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_library(tinyxml2 tinyxml2/tinyxml2.cpp)
add_executable(testTinyxml2 src/testTinyxml.cpp)
target_link_libraries(testTinyxml2 tinyxml2) 

# find_library(CERES_LIBRARY NAMES ceres PATHS "/usr/local/lib/libceres.a")


rosidl_generate_interfaces(${PROJECT_NAME} "msg/CloudInfo.msg" "msg/FusionDataFrame.msg" "msg/FusionPoints.msg" "msg/OdomMsg.msg" "srv/SaveMap.srv" DEPENDENCIES geometry_msgs std_msgs nav_msgs sensor_msgs)
ament_export_dependencies(rosidl_default_runtime)

add_executable(${PROJECT_NAME}_transformFusion src/transformFusion.cpp)
ament_target_dependencies(${PROJECT_NAME}_transformFusion rclcpp rclpy std_msgs  gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM)
rosidl_target_interfaces(${PROJECT_NAME}_transformFusion ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_transformFusion "${cpp_typesupport_target}") 

add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
ament_target_dependencies(${PROJECT_NAME}_imageProjection rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL)
rosidl_target_interfaces(${PROJECT_NAME}_imageProjection ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_imageProjection ${PCL_LIBRARIES}  "${cpp_typesupport_target}") 

add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
ament_target_dependencies(${PROJECT_NAME}_imuPreintegration rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM Eigen)
target_link_libraries(${PROJECT_NAME}_imuPreintegration ${PCL_LIBRARIES} gtsam ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_mapOptimization src/mapOptmization.cpp)
ament_target_dependencies(${PROJECT_NAME}_mapOptimization rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs ndt_omp visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM)
rosidl_target_interfaces(${PROJECT_NAME}_mapOptimization ${PROJECT_NAME} "rosidl_typesupport_cpp")
if (OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}_mapOptimization ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}" OpenMP::OpenMP_CXX)
else()
  target_link_libraries(${PROJECT_NAME}_mapOptimization ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")
endif()

add_executable(${PROJECT_NAME}_transFusionData src/transFusionData.cpp)
ament_target_dependencies(${PROJECT_NAME}_transFusionData rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM Eigen)
rosidl_target_interfaces(${PROJECT_NAME}_transFusionData ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_transFusionData ${PCL_LIBRARIES}  "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_slotPointsMapping src/slotPointsMapping.cpp)
ament_target_dependencies(${PROJECT_NAME}_slotPointsMapping rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM Eigen)
rosidl_target_interfaces(${PROJECT_NAME}_slotPointsMapping ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_slotPointsMapping ${PCL_LIBRARIES}  ${OpenMP_CXX_FLAGS} gtsam ${GeographicLib_LIBRARIES} tinyxml2 "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_simpleGPSOdom src/simpleGPSOdom.cpp)
ament_target_dependencies(${PROJECT_NAME}_simpleGPSOdom rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs PCL GTSAM Eigen)
rosidl_target_interfaces(${PROJECT_NAME}_simpleGPSOdom ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_simpleGPSOdom ${PCL_LIBRARIES}  ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")

add_executable(vcuImuOdom src/WheelOdom/vcuImuOdom.cpp)
ament_target_dependencies(vcuImuOdom rclcpp rclpy std_msgs gpal_msgs radar_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
rosidl_target_interfaces(vcuImuOdom ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(vcuImuOdom  ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")


# add_executable(${PROJECT_NAME}_imuGPSOptimization src/imuGPSOptimization.cpp)
# ament_target_dependencies(${PROJECT_NAME}_imuGPSOptimization rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
# rosidl_target_interfaces(${PROJECT_NAME}_imuGPSOptimization ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_imuGPSOptimization ${PCL_LIBRARIES} gtsam ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_vcuGPSOptimization src/vcuGPSOptimization.cpp  src/utmConverter/utm.cpp src/utmConverter/tranmerc.cpp )
ament_target_dependencies(${PROJECT_NAME}_vcuGPSOptimization rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
rosidl_target_interfaces(${PROJECT_NAME}_vcuGPSOptimization ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_vcuGPSOptimization ${PCL_LIBRARIES} gtsam  ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}" )

# add_executable(${PROJECT_NAME}_writePath src/writePath.cpp)
# ament_target_dependencies(${PROJECT_NAME}_writePath rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
# rosidl_target_interfaces(${PROJECT_NAME}_writePath ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_writePath ${PCL_LIBRARIES}  "${cpp_typesupport_target}")

# add_executable(${PROJECT_NAME}_imuConverter src/imuConverter/imuConverter.cpp)
# ament_target_dependencies(${PROJECT_NAME}_imuConverter rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
# rosidl_target_interfaces(${PROJECT_NAME}_imuConverter ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_imuConverter   "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_parkSlotPointCloud src/parkSlot/parkSlotPointCloud.cpp)
ament_target_dependencies(${PROJECT_NAME}_parkSlotPointCloud rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
rosidl_target_interfaces(${PROJECT_NAME}_parkSlotPointCloud ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_parkSlotPointCloud ${PCL_LIBRARIES}   "${cpp_typesupport_target}")


# add_executable(${PROJECT_NAME}_vcuGpsDrivingMapping src/vcuGpsDrivingMapping.cpp)
# ament_target_dependencies(${PROJECT_NAME}_vcuGpsDrivingMapping rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen)
# rosidl_target_interfaces(${PROJECT_NAME}_vcuGpsDrivingMapping ${PROJECT_NAME} "rosidl_typesupport_cpp")
# target_link_libraries(${PROJECT_NAME}_vcuGpsDrivingMapping ${PCL_LIBRARIES}  ${OpenMP_CXX_FLAGS} gtsam ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")

add_executable(${PROJECT_NAME}_slotsLocalization src/slotsLocal.cpp)
ament_target_dependencies(${PROJECT_NAME}_slotsLocalization rclcpp rclpy std_msgs gpal_msgs radar_msgs gpal_vision_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs  Eigen CERES)
rosidl_target_interfaces(${PROJECT_NAME}_slotsLocalization ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_slotsLocalization ${PCL_LIBRARIES}  ${OpenMP_CXX_FLAGS}  ${GeographicLib_LIBRARIES} "${cpp_typesupport_target}")


install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

install(
  TARGETS ${PROJECT_NAME}_imageProjection
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_imuPreintegration
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_transformFusion
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_mapOptimization
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_transFusionData
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_simpleGPSOdom
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}_slotPointsMapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS vcuImuOdom
  DESTINATION lib/${PROJECT_NAME}
)

# install( 
#   TARGETS ${PROJECT_NAME}_imuGPSOptimization
#   DESTINATION lib/${PROJECT_NAME}
# )

install( 
  TARGETS ${PROJECT_NAME}_vcuGPSOptimization
  DESTINATION lib/${PROJECT_NAME}
)

# install( 
#   TARGETS ${PROJECT_NAME}_writePath
#   DESTINATION lib/${PROJECT_NAME}
# )

# install( 
#   TARGETS ${PROJECT_NAME}_imuConverter
#   DESTINATION lib/${PROJECT_NAME}
# )

install(
  TARGETS ${PROJECT_NAME}_parkSlotPointCloud
  DESTINATION lib/${PROJECT_NAME}
)

# install(
#   TARGETS testTinyxml2
#   DESTINATION lib/${PROJECT_NAME}
# )

# install(
#   TARGETS utmConverter
#   DESTINATION lib/${PROJECT_NAME}
# )


# install( 
#   TARGETS ${PROJECT_NAME}_vcuGpsDrivingMapping
#   DESTINATION lib/${PROJECT_NAME}
# )

install(
  TARGETS ${PROJECT_NAME}_slotsLocalization
  DESTINATION lib/${PROJECT_NAME}
)


ament_export_include_directories(include)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  ament_lint_auto_find_test_dependencies()
endif()
ament_package()
