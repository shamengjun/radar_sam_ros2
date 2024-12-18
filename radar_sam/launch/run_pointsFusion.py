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
    rviz_config_file = os.path.join(share_dir, 'config', 'gps_imu.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    print("urdf_file_name : {}".format(xacro_path))

    launch_file =  IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('radar_sam'), 'launch'),
         '/run_vcu_gps_odom.py'])
      )

    return LaunchDescription([
        # module_navsat_transform_node,
        params_declare,
        launch_file,
        Node(
            package='radar_sam',
            executable='radar_sam_pointsFusion',
            name='radar_sam_pointsFusion',
            parameters=[parameter_file],
            output='screen',
            prefix=['xterm -e gdb --args']
        ),
        Node(
            package='radar_sam',
            executable='radar_sam_transFusionData',
            name='radar_sam_transFusionData',
            parameters=[parameter_file],
            output='screen'
        )
    ])
    