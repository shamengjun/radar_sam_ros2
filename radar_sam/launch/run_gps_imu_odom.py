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
    rviz_config_file = os.path.join(share_dir, 'config', 'pointsFusion.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            share_dir, 'config', 'params.yaml'),
        description='FPath to the ROS2 parameters file to use.')

    print("urdf_file_name : {}".format(xacro_path))

   
   
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
            # Node(
            #     package='tf2_ros',
            #     executable='static_transform_publisher',
            #     arguments='--frame-id odom --child-frame-id gps'.split(' '),
            #     parameters=[parameter_file],
            #     output='screen'
            #     ),
            
            Node(
                package='radar_sam',
                executable='radar_sam_simpleGPSOdom',
                name='radar_sam_simpleGPSOdom',
                parameters=[parameter_file],
                output='screen'
                # prefix=['xterm -e gdb --args']
            ),
            Node(
                package='radar_sam',
                executable='radar_sam_imuGPSOptimization',
                name='radar_sam_imuGPSOptimization',
                parameters=[parameter_file],
                output='screen'
                # respawn=True,
                # prefix=['xterm -e gdb --args']
            ),
            # Node(
            #     package='radar_sam',
            #     executable='vcuImuOdom',
            #     name='vcuImuOdom',
            #     parameters=[parameter_file],
            #     output='screen'
            # ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_file],
                output='screen'
            )
    ])
   
    
