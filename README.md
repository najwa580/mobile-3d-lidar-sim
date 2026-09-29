# ROS2 Mobile Robot Navigation using 2D LiDAR and DQN

Simulate and control a mobile robot using ROS2 Humble and Gazebo Classic. This project uses a 2D LiDAR sensor for environment perception and Deep Q-Network (DQN) for autonomous navigation. A Double DQN implementation is also included for comparison and improvement of the navigation policy.

![Robot in Gazebo Simulation](img/robot_gz_sim.png)

![Robot Navigation in RViz](img/rviz_navigation.png)

## Prerequisites

* Ubuntu 22.04
* ROS2 Humble (tested)
* Gazebo Classic (tested)
* RViz2
* Python 3
* PyTorch

## Installation and Setup

1. **Source ROS2 environment:**

   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Create a ROS2 workspace:**

   ```bash
   mkdir -p ~/robot/src
   cd ~/robot/src
   ```

3. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   ```

4. **Install dependencies:**

   ```bash
   cd ~/robot
   rosdep install --from-paths src --ignore-src -r -y
   ```

5. **Build the workspace:**

   ```bash
   cd ~/robot
   colcon build --symlink-install
   ```

6. **Source the workspace setup file:**

   ```bash
   source install/setup.bash
   ```

## Running the Simulation

1. **Launch the robot in Gazebo:**

   To run the default simulation:

   ```bash
   ros2 launch my_bot launch_sim.launch.py
   ```

   To run the simulation with a custom world:

   ```bash
   ros2 launch my_bot launch_sim.launch.py world:=./src/mobile-3d-lidar-sim/my_bot/worlds/obstacles.world
   ```

2. **Control the robot manually:**

   In a new terminal, run the following command:

   ```bash
   ros2 run teleop_twist_keyboard teleop_twist_keyboard
   ```

3. **Check the 2D LiDAR data:**

   The LiDAR measurements are published on the `/scan` topic:

   ```bash
   ros2 topic echo /scan
   ```

4. **Visualize the simulation in RViz:**

   In a third terminal, run:

   ```bash
   rviz2
   ```

   * **Add visualization elements:**

     * Add a `RobotModel`.
     * Add a `LaserScan`.

   * **Configure visualization:**

     * In `Global Options`, change `Fixed Frame` to `odom`.
     * In `RobotModel`, use `/robot_description`.
     * In `LaserScan`, set the `Topic` to `/scan`.

## DQN Navigation

The project uses a Deep Q-Network (DQN) to learn an autonomous navigation policy.

The DQN receives information from the 2D LiDAR and the target position, then selects one of six possible actions to control the robot.

The main navigation loop is:

```text
2D LiDAR + Goal
       |
       v
     State
       |
       v
      DQN
       |
       v
     Action
       |
       v
    /cmd_vel
       |
       v
     Robot
```

To train the DQN model:

```bash
ros2 run my_bot train_dqn
```

The trained model can then be used for autonomous navigation.

## Double DQN

A Double DQN implementation is also included in the project.

Double DQN uses an online network to select the action and a target network to evaluate it. It is used to reduce the overestimation of Q-values observed with the standard DQN.

The Double DQN implementation is mainly used to evaluate and compare the navigation performance with the standard DQN.

## Real Robot

The trained DQN model can also be deployed on the real mobile robot.

The real robot uses:

* Jetson Orin Nano
* YDLIDAR X4 2D LiDAR
* Raspberry Pi Pico with micro-ROS
* BTS7960 motor drivers
* DC geared motors

The communication chain is:

```text
YDLIDAR X4
     |
     v
   /scan
     |
     v
Jetson Orin Nano
     |
     v
    DQN
     |
     v
 /cmd_vel
     |
     v
Raspberry Pi Pico
     |
     v
BTS7960
     |
     v
DC Motors
```

## ROS2 Topics

The main ROS2 topics used by the project are:

* `/scan` — 2D LiDAR measurements.
* `/odom` — robot odometry.
* `/cmd_vel` — velocity commands.
* `/tf` — coordinate transformations.
* `/robot_description` — robot model description.

## Additional Information

This project provides a complete environment for testing autonomous mobile robot navigation using ROS2, Gazebo Classic, 2D LiDAR and Deep Reinforcement Learning.

The DQN agent learns to navigate toward a target while avoiding obstacles. The trained model can be evaluated in simulation and deployed on a real mobile robot.

The project also includes a Double DQN implementation for comparison with the standard DQN approach.

## License

This project is licensed under the Apache License 2.0.

See the `LICENSE` file for more information.

