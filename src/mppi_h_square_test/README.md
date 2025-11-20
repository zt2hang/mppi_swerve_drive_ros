# MPPI_H Circular Path Tracking

This project implements a circular path tracking algorithm using the MPPI_H (Model Predictive Path Integral Control) method for differential drive robots. The goal is to control the robot to follow a predefined circular trajectory while maintaining stability and responsiveness.

## Project Structure

- **scripts/circular_path_tracking_mppi_h.py**: This script contains the implementation of the circular path tracking using the MPPI_H algorithm. It includes functionalities for path generation, state propagation, control input computation, and visualization.

- **config/mppi_h_params.yaml**: This configuration file contains parameters for the MPPI_H algorithm, such as control gains, prediction horizon, and weight matrices. These parameters are crucial for tuning the performance and responsiveness of the algorithm.

- **config/track.yaml**: This file defines parameters related to the tracking path, including the radius of the circular path and speed limits. These parameters are used to generate the reference trajectory for the robot.

- **launch/mppi_h_circular.launch**: This is a ROS launch file that starts the MPPI_H circular path tracking node. It configures the node parameters and dependencies to run within the ROS environment.

- **CMakeLists.txt**: This file contains the build configuration for CMake, defining the build rules, dependencies, and target settings for the project.

- **package.xml**: This is the ROS package descriptor file that includes the package name, version, maintainer information, and dependencies.

## Usage

1. **Setup**: Ensure that you have ROS installed and properly configured on your system.

2. **Build the Package**: Navigate to the root of the project directory and build the package using the following command:
   ```
   catkin_make
   ```

3. **Launch the Node**: After building the package, you can launch the MPPI_H circular path tracking node using:
   ```
   roslaunch mppi_h_circular_test mppi_h_circular.launch
   ```

4. **Visualization**: The script includes visualization capabilities to monitor the robot's trajectory and control inputs in real-time.

## Dependencies

- ROS (Robot Operating System)
- Python 3
- NumPy
- Matplotlib
- Other ROS packages as specified in `package.xml`

## Author

- [Your Name] - [Your Contact Information]

## License

This project is licensed under the MIT License - see the LICENSE file for details.