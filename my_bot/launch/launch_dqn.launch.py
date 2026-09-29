from ament_index_python.packages import get_package_share_directory
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # =========================================================
    # RSP — Robot State Publisher
    # Lit my_robot.urdf.xacro et publie sur /robot_description
    # =========================================================
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('my_bot'),
                'launch',
                'rsp.launch.py'
            )
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # =========================================================
    # GAZEBO
    # =========================================================
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={
            'world': os.path.join(
                get_package_share_directory('my_bot'),
                'worlds',
                'obstacles.world'
            )
        }.items()
    )

    # =========================================================
    # SPAWN — via -topic robot_description (xacro compilé RSP)
    #
    # ❌ ANCIEN (cassé) : '-file', robot_urdf
    #    → cherche un .urdf compilé qui n'existe pas dans /urdf/
    #
    # ✅ CORRECT : '-topic', 'robot_description'
    #    → utilise le URDF compilé en live par RSP depuis le xacro
    #
    # Position spawn : (-7, -7, 0.10)
    #   z = base_z_footprint + wheel_radius = 0.03 + 0.065 ≈ 0.10m
    #   yaw = 0.785 rad (45°) → robot orienté vers goal (5, 6)
    # =========================================================
    spawn_entity = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'my_robot',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '0.10',
                    '-R', '0.0',
                    '-P', '0.0',
                    '-Y', '0.785',
                ],
                output='screen'
            )
        ]
    )

    # =========================================================
    # DQN TRAINING NODE — délai pour laisser Gazebo stabiliser
    # =========================================================
    train_dqn_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='my_bot',
                executable='train_dqn',
                name='train_dqn_node',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        train_dqn_node,
    ])
