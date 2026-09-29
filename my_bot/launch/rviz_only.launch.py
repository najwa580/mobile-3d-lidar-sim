#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    # 1. Trouver le dossier du package my_bot
    pkg_share = FindPackageShare('my_bot').find('my_bot')

    # 2. Chemin vers le fichier Xacro principal
    xacro_file = os.path.join(pkg_share, 'description', 'my_robot.urdf.xacro')

    # 3. Générer le URDF à partir du xacro
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        xacro_file
    ])

    # 4. Paramètre robot_description (passé à robot_state_publisher)
    #    ParameterValue(..., value_type=str) force ROS2 à traiter le contenu
    #    comme une chaîne brute au lieu d'essayer de le parser en YAML
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # 5. Chemin config RViz (décommente une fois que tu l'auras créée/sauvegardée)
    rviz_config = os.path.join(pkg_share, 'config', 'navigation.rviz')

    return LaunchDescription([
        # Publie le modèle URDF -> nécessaire pour RViz
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[robot_description],
        ),

        # Permet de bouger les roues/joints manuellement pour vérifier le modèle
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # Lance RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
