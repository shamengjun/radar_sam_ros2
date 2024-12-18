import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    share_dir = get_package_share_directory('radar_sam')
    parameter_file = LaunchConfiguration('params_file')
    xacro_path = os.path.join(share_dir, 'config', 'robot.urdf.xacro')
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    print("urdf_file_name : {}".format(xacro_path))

    # module_navsat_transform_node = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         os.path.join(get_package_share_directory('navsat_transform'), 'launch'),
    #         '/navsat_transform_launch.py'])
    # )
    use_gps_factor = True #True
    if(not use_gps_factor):
        return LaunchDescription([
            # module_navsat_transform_node,
            params_declare,
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
                parameters=[parameter_file],
                output='screen'
                ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{
                    'robot_description': Command(['xacro', ' ', xacro_path])
                }]
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imuConvert',
                name='radar_sam_imuConvert',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_transFusionData',
                name='radar_sam_transFusionData',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imuPreintegration',
                name='radar_sam_imuPreintegration',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imageProjection',
                name='radar_sam_imageProjection',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_transformFusion',
                name='radar_sam_transformFusion',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_mapOptimization',
                name='radar_sam_mapOptimization',
                parameters=[parameter_file],
                output='screen'
                # prefix=['xterm -e gdb --args']
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='vcuImuOdom',
                name='vcuImuOdom',
                parameters=[parameter_file],
                output='screen'
            )
        ])
    else:
        return LaunchDescription([
            # module_navsat_transform_node,
            params_declare,
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments='0.0 0.0 0.0 0.0 0.0 0.0 map odom'.split(' '),
                parameters=[parameter_file],
                output='screen'
                ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments='0.0 0.0 0.0 0.0 0.0 0.0 base_link imu'.split(' '),
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments='0.0 0.0 0.0 0.0 0.0 0.0 base_link gps'.split(' '),
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{
                    'robot_description': Command(['xacro', ' ', xacro_path])
                }]
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_transFusionData',
                name='radar_sam_transFusionData',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_gps',
                parameters=[os.path.join(get_package_share_directory("radar_sam"), 'config', 'ekf_gps.yaml')],
                output='screen'
            ),
            Node(
                package='robot_localization',
                executable='navsat_transform_node',
                name='navsat',
                remappings = [("imu/data","/imu"),("gps/fix","/gps")],
                parameters=[os.path.join(get_package_share_directory("radar_sam"), 'config', 'navsat.yaml')],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imuConvert',
                name='radar_sam_imuConvert',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imuPreintegration',
                name='radar_sam_imuPreintegration',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imageProjection',
                name='radar_sam_imageProjection',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_transformFusion',
                name='radar_sam_transformFusion',
                parameters=[parameter_file],
                output='screen'
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_mapOptimization',
                name='radar_sam_mapOptimization',
                parameters=[parameter_file],
                output='screen',
                prefix=['xterm -e gdb --args']
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                output='screen'
            )
        ])
    
