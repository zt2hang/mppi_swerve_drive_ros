#!/home/zzt/planner_code/mppi_swerve_drive_ros/mppi_mlp_venv/bin/python3
import rospy
import torch
import numpy as np
# import tf
import sys
import os
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32
from scipy.signal import savgol_filter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from online_residual_learner import OnlineResidualLearner

def get_yaw_from_quaternion(q):
    """
    Calculate yaw from quaternion (x, y, z, w)
    """
    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

class FourWheelSteeringKinematics:
    def __init__(self, l_f=0.5, l_r=0.5, d_l=0.5, d_r=0.5, radius=0.2):
        self.l_f = l_f
        self.l_r = l_r
        self.d_l = d_l
        self.d_r = d_r
        self.radius = radius
        
        # Wheel positions relative to center [x, y]
        # FL, FR, RL, RR
        self.wheel_pos = torch.tensor([
            [l_f, d_l],
            [l_f, -d_r],
            [-l_r, d_l],
            [-l_r, -d_r]
        ], dtype=torch.float32)

    def inverse_kinematics_batch(self, vx, vy, omega):
        """
        Batch Inverse Kinematics
        Args:
            vx, vy, omega: Tensors of shape (K,) or (K, 1)
        Returns:
            wheel_vels: (K, 4)
            steer_angles: (K, 4)
        """
        K = vx.shape[0]
        device = vx.device
        
        # Ensure inputs are (K, 1)
        vx = vx.view(K, 1)
        vy = vy.view(K, 1)
        omega = omega.view(K, 1)
        
        # Wheel positions (4, 2) -> (1, 4, 2)
        pos = self.wheel_pos.to(device).unsqueeze(0).expand(K, -1, -1)
        
        # V_wheel_x = vx - omega * y
        # V_wheel_y = vy + omega * x
        # Note: pos[..., 0] is x, pos[..., 1] is y
        
        v_wx = vx - omega * pos[..., 1] # (K, 4)
        v_wy = vy + omega * pos[..., 0] # (K, 4)
        
        wheel_vels = torch.sqrt(v_wx**2 + v_wy**2)
        steer_angles = torch.atan2(v_wy, v_wx)
        
        return wheel_vels, steer_angles

class MPPIControllerNode:
    def __init__(self):
        rospy.init_node('mppi_python_node')
        
        # Parameters
        self.horizon = rospy.get_param('~horizon', 30)
        self.dt = rospy.get_param('~dt', 0.05)
        self.num_samples = rospy.get_param('~num_samples', 2000)
        self.lambda_param = rospy.get_param('~lambda', 10.0) # Reduced lambda for smoother mixing
        self.noise_sigma = torch.tensor([0.3, 0.3, 0.5], device='cuda') # Reduced noise
        self.target_velocity = 1.6 # Target speed [m/s]
        
        # Cost Weights
        self.w_dist = 20.0
        self.w_vel = 10.0
        self.w_smooth = 20.0 # Increased from 5.0 to 20.0 to reduce jitter
        
        # Kinematics Params
        self.kinematics = FourWheelSteeringKinematics()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"MPPI running on {self.device}")
        
        # Learner
        self.learner = OnlineResidualLearner(
            input_dim=11,
            output_dim=3,
            hidden_dim=32,
            device=self.device.type
        )
        self.train_steps = 0
        self.mlp_weight = 0.0 # Warm-up weight for MLP
        
        # State
        self.current_state = None # [x, y, yaw, vx, vy, omega]
        self.reference_path = None
        
        # MPPI Initialization
        self.U = torch.zeros(self.horizon, 3, device=self.device) # Mean control inputs [vx, vy, omega]
        
        # ROS Interfaces
        self.sub_odom = rospy.Subscriber('/groundtruth_odom', Odometry, self.odom_callback)
        self.sub_path = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_callback)
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Timer for Control Loop (20Hz)
        rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        # Timer for Training (10Hz)
        rospy.Timer(rospy.Duration(0.1), self.train_loop)

    def odom_callback(self, msg):
        # Extract state
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw = get_yaw_from_quaternion(q)
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z
        
        # Update current state
        self.current_state = np.array([x, y, yaw, vx, vy, omega])

        # If we have a previous prediction, we can add data to learner
        self.update_learner_data()

    def path_callback(self, msg):
        # Convert path to numpy array for fast lookup
        path_points = []
        for pose in msg.poses:
            path_points.append([pose.pose.position.x, pose.pose.position.y])
        self.reference_path = torch.tensor(path_points, device=self.device, dtype=torch.float32)

    def get_kinematic_next_state(self, state, control):
        """
        Simple Kinematic Model (CPU/Numpy for single step verification)
        state: [x, y, yaw, vx, vy, w]
        control: [vx_cmd, vy_cmd, w_cmd]
        """
        dt = self.dt
        # Assume perfect tracking of velocity command for the base kinematic model
        # x_next = x + (vx_cmd * cos(yaw) - vy_cmd * sin(yaw)) * dt
        # y_next = y + (vx_cmd * sin(yaw) + vy_cmd * cos(yaw)) * dt
        # yaw_next = yaw + w_cmd * dt
        
        yaw = state[2]
        vx_cmd, vy_cmd, w_cmd = control
        
        dx = (vx_cmd * np.cos(yaw) - vy_cmd * np.sin(yaw)) * dt
        dy = (vx_cmd * np.sin(yaw) + vy_cmd * np.cos(yaw)) * dt
        dyaw = w_cmd * dt
        
        return np.array([state[0]+dx, state[1]+dy, state[2]+dyaw, vx_cmd, vy_cmd, w_cmd])

    def train_loop(self, event):
        # Perform one training step
        loss = self.learner.train_step()
        if loss > 0:
            self.train_steps += 1
            # Warm-up: linearly increase weight from 0.0 to 1.0 over first 200 steps
            self.mlp_weight = min(1.0, self.train_steps / 200.0)
            rospy.loginfo_throttle(5.0, f"Training Loss: {loss:.5f}, MLP Weight: {self.mlp_weight:.2f}")

    def control_loop(self, event):
        if self.current_state is None:
            rospy.loginfo_throttle(2.0, "Waiting for Odometry...")
            return
        if self.reference_path is None:
            rospy.loginfo_throttle(2.0, "Waiting for Path...")
            return

        # 1. Shift Control Sequence
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = torch.zeros(3, device=self.device)

        # 2. Sample Noise
        noise = torch.randn(self.num_samples, self.horizon, 3, device=self.device) * self.noise_sigma
        
        # 3. Perturbed Controls
        # U_perturbed: (K, H, 3)
        U_perturbed = self.U.unsqueeze(0) + noise
        
        # 4. Rollout Trajectories
        costs = torch.zeros(self.num_samples, device=self.device)
        
        # Initial State (K, 3) -> x, y, yaw
        curr_x = torch.tensor(self.current_state[0], device=self.device).repeat(self.num_samples)
        curr_y = torch.tensor(self.current_state[1], device=self.device).repeat(self.num_samples)
        curr_yaw = torch.tensor(self.current_state[2], device=self.device).repeat(self.num_samples)
        
        # Loop over horizon
        prev_u = torch.zeros(self.num_samples, 3, device=self.device) # For smoothness cost
        
        for t in range(self.horizon):
            # Control at time t: (K, 3)
            u_t = U_perturbed[:, t, :] # vx, vy, omega
            
            # --- INTEGRATE LEARNER HERE ---
            # 1. Get Kinematic Control (Wheel Vels & Angles) for MLP Input
            #    We need to convert body velocity commands to wheel commands
            wheel_vels, steer_angles = self.kinematics.inverse_kinematics_batch(u_t[:, 0], u_t[:, 1], u_t[:, 2])
            
            # 2. Prepare MLP Input
            #    State: Current Body Velocity (We approximate this as the Command for the rollout, 
            #           or use the previous step's velocity. Here we use command u_t as proxy for current vel)
            #    Action: 8-dim wheel commands
            #    Input shape: (K, 11)
            
            # Flatten wheel/steer for concatenation: (K, 4) -> (K, 4)
            # Concatenate: [vx, vy, w, v_fl, v_fr..., th_fl...]
            
            # Note: In a real dynamic rollout, 'state' should be the actual velocity. 
            # Since we are doing kinematic rollout, we assume velocity tracks command instantly,
            # BUT the residual learner predicts the ERROR in that assumption.
            
            # Predict Residual
            # Input: State (3) + Action (8)
            # We use u_t as 'current velocity' approximation for the input
            residuals = self.learner.predict_batch(u_t, torch.cat([wheel_vels, steer_angles], dim=1))
            
            if t == 0:
                # Log the mean residual for the first step of the rollout
                mean_res = residuals.mean(dim=0)
                rospy.loginfo_throttle(2.0, f"MLP Pred (Avg): dx={mean_res[0]:.4f}, dy={mean_res[1]:.4f}, dw={mean_res[2]:.4f}")
            
            # 3. Apply Residual (Weighted by warm-up factor)
            # Effective Velocity = Command + Weight * Residual
            v_eff = u_t + self.mlp_weight * residuals
            
            # --- KINEMATIC UPDATE WITH RESIDUAL ---
            # x_next = x + (v_eff_x * cos(yaw) - v_eff_y * sin(yaw)) * dt
            
            vx_eff = v_eff[:, 0]
            vy_eff = v_eff[:, 1]
            w_eff = v_eff[:, 2]
            
            dx = (vx_eff * torch.cos(curr_yaw) - vy_eff * torch.sin(curr_yaw)) * self.dt
            dy = (vx_eff * torch.sin(curr_yaw) + vy_eff * torch.cos(curr_yaw)) * self.dt
            dyaw = w_eff * self.dt
            
            curr_x += dx
            curr_y += dy
            curr_yaw += dyaw
            
            # --- CALCULATE COST ---
            # 1. Distance Cost
            if len(self.reference_path) > 0:
                # Take a subset of path points to speed up
                # (K, 2)
                pos = torch.stack([curr_x, curr_y], dim=1)
                
                # We use a coarse path (every 10th point)
                coarse_path = self.reference_path[::5].to(dtype=torch.float32)
                pos = pos.to(dtype=torch.float32)
                dists = torch.cdist(pos, coarse_path) # (K, M)
                min_dists, _ = torch.min(dists, dim=1)
                costs += min_dists * self.w_dist
                
                # Add Terminal Cost (Distance to end of local path)
                if t == self.horizon - 1:
                    end_pos = self.reference_path[-1].unsqueeze(0).to(dtype=torch.float32) # (1, 2)
                    term_dist = torch.norm(pos - end_pos, dim=1)
                    costs += term_dist * 50.0 # Terminal weight
            
            # 2. Velocity Tracking Cost
            # Encourage moving at target velocity
            vel_error = (u_t[:, 0] - self.target_velocity)**2
            costs += vel_error * self.w_vel
            
            # 3. Smoothness Cost (Control Change)
            if t > 0:
                control_change = torch.sum((u_t - prev_u)**2, dim=1)
                costs += control_change * self.w_smooth
            
            prev_u = u_t
        

        # 5. Update Control
        # Weights
        min_cost = torch.min(costs)
        exp_costs = torch.exp(-1.0 / self.lambda_param * (costs - min_cost))
        weights = exp_costs / torch.sum(exp_costs)
        
        # Weighted Sum
        # (K, H, 3) * (K, 1, 1) -> sum over K -> (H, 3)
        weights_expanded = weights.view(-1, 1, 1)
        self.U += torch.sum(weights_expanded * noise, dim=0)
        
        # --- Savitzky-Golay Filter for Smoothness ---
        # Apply filter to the updated control sequence self.U
        try:
            u_np = self.U.cpu().numpy()
            # Window size must be odd and <= horizon
            window_size = min(11, self.horizon if self.horizon % 2 != 0 else self.horizon - 1)
            poly_order = 3
            
            if window_size > poly_order:
                # Filter each dimension: vx, vy, omega
                u_np[:, 0] = savgol_filter(u_np[:, 0], window_size, poly_order)
                u_np[:, 1] = savgol_filter(u_np[:, 1], window_size, poly_order)
                u_np[:, 2] = savgol_filter(u_np[:, 2], window_size, poly_order)
                
                self.U = torch.from_numpy(u_np).to(self.device)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"SG Filter failed: {e}")

        # 6. Publish Command
        # Use the first control input
        optimal_u = self.U[0].cpu().numpy()
        
        # Debug print
        rospy.loginfo_throttle(1.0, f"Optimal Control: vx={optimal_u[0]:.2f}, vy={optimal_u[1]:.2f}, w={optimal_u[2]:.2f}")

        cmd = Twist()
        cmd.linear.x = float(optimal_u[0])
        cmd.linear.y = float(optimal_u[1])
        cmd.angular.z = float(optimal_u[2])
        self.pub_cmd.publish(cmd)
        
        # 7. Data Collection for Learner
        # We need to store (State_t, Action_t, Target_Residual_t)
        # State_t: The state at the BEGINNING of this control step (self.current_state)
        # Action_t: The command we just sent (optimal_u)
        # Target_Residual_t: We can't know this until the NEXT callback!
        
        # So we store 'pending' data
        self.last_training_state = self.current_state.copy()
        self.last_training_action = optimal_u.copy()
        
        # Calculate Kinematic Prediction for the NEXT state
        # We will compare this with the ACTUAL next state in the next odom callback
        self.last_kinematic_next = self.get_kinematic_next_state(self.current_state, optimal_u)

    def update_learner_data(self):
        # Called in odom callback or separate loop
        # If we have pending data
        if hasattr(self, 'last_training_state') and self.current_state is not None:
            # Actual Next State is self.current_state
            # Kinematic Next State is self.last_kinematic_next
            
            # Calculate Velocity Residual
            # We are learning the error in VELOCITY, not Position
            # So we compare Actual Velocity vs Command Velocity (assuming kinematic model says Vel = Cmd)
            
            # Actual Vel
            v_actual = self.current_state[3:6] # vx, vy, w
            
            # Expected Vel (Kinematic)
            # In simple kinematic model, Expected Vel = Command Vel
            v_expected = self.last_training_action # vx_cmd, vy_cmd, w_cmd
            
            # Residual = Actual - Expected
            residual = v_actual - v_expected
            
            rospy.loginfo_throttle(2.0, f"Target Residual (Err): dx={residual[0]:.4f}, dy={residual[1]:.4f}, dw={residual[2]:.4f}")
            
            # Add to buffer
            # Input State: Body Vel (from last step)
            input_vel = self.last_training_state[3:6]
            
            # Input Action: Wheel Commands (from last step)
            # We need to re-calculate the wheel commands we sent
            # (We could have stored them, but recalculating is fine)
            vx, vy, w = self.last_training_action
            # Convert to tensor for IK
            t_vx = torch.tensor([vx], device=self.device)
            t_vy = torch.tensor([vy], device=self.device)
            t_w = torch.tensor([w], device=self.device)
            w_vels, steers = self.kinematics.inverse_kinematics_batch(t_vx, t_vy, t_w)
            
            # Flatten
            cmd_input = torch.cat([w_vels.view(-1), steers.view(-1)]).cpu().numpy()
            
            self.learner.add_observation(input_vel, cmd_input, residual)

if __name__ == '__main__':
    try:
        node = MPPIControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
