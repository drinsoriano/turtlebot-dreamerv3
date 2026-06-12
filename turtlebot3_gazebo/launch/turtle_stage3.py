import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    _launch_dir = os.path.dirname(os.path.realpath(__file__))
    _pkg_dir    = os.path.dirname(_launch_dir)

    world_file_name = 'turtlebot3_dqn_stage3/' + TURTLEBOT3_MODEL + '.model'
    world           = os.path.join(_pkg_dir, 'worlds', world_file_name)
    pkg_gazebo_ros  = get_package_share_directory('gazebo_ros')

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='launch the Gazebo GUI client; gui:=false for headless '
                        '(no gzclient OpenGL load — avoids eGPU/Thunderbolt drops)',
        ),
        SetEnvironmentVariable(
            'GAZEBO_MODEL_PATH',
            os.path.join(_pkg_dir, 'models') + ':' + os.environ.get('GAZEBO_MODEL_PATH', ''),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
            condition=IfCondition(LaunchConfiguration('gui')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(_launch_dir, 'robot_state_publisher.launch.py')
            ),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ])
