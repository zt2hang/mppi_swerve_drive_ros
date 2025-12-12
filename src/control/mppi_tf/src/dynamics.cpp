/**
 * @file dynamics.cpp
 * @brief Implementation of force-based dynamics model
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/dynamics.hpp"
#include <cmath>

namespace mppi_tf
{

DynamicsModel::DynamicsModel(const VehicleParams& vehicle, DynamicsMode mode)
    : params_(vehicle)
    , mode_(mode)
    , tire_model_(vehicle.tire)
    , force_allocator_(vehicle.geometry, vehicle.mass, AllocationParams{})
    , C_alpha_current_(vehicle.tire.C_alpha)
    , mu_current_(vehicle.tire.mu_peak)
{
}

void DynamicsModel::setTireParams(double C_alpha, double mu)
{
    C_alpha_current_ = C_alpha;
    mu_current_ = mu;
    tire_model_.setCorneringStiffness(C_alpha);
    tire_model_.setFrictionCoeff(mu);
    force_allocator_.setFrictionCoeff(mu);
}

// ============================================================================
// Force-Based State Propagation
// ============================================================================

FullState DynamicsModel::stepForce(
    const FullState& state, 
    const BodyForce& force, 
    double dt) const
{
    // Clamp force to limits
    BodyForce clamped_force = clampForce(force);
    
    // Apply inertial corrections (Coriolis terms in body frame)
    BodyForce effective_force = applyInertialCorrections(clamped_force, state.velocity());
    
    // Integrate acceleration to velocity
    BodyVelocity new_vel = integrateAcceleration(state.velocity(), effective_force, dt);
    
    // Clamp velocity
    new_vel.clamp(params_.limits.vx_max, params_.limits.vy_max, params_.limits.omega_max);
    
    // Integrate velocity to pose
    Pose2D new_pose = integratePose(state.pose(), new_vel, dt);
    
    // Construct new state
    FullState next;
    next.setPose(new_pose);
    next.setVelocity(new_vel);
    next.normalizeYaw();
    
    return next;
}

std::pair<FullState, BodyForce> DynamicsModel::stepTireModel(
    const FullState& state,
    const BodyForce& desired_force,
    double dt) const
{
    // Allocate desired force to wheels
    auto alloc_result = force_allocator_.allocate(desired_force);
    
    // The achieved force may differ from desired due to friction limits
    BodyForce achieved_force = alloc_result.achieved_force;
    
    // Propagate state with achieved force
    FullState next = stepForce(state, achieved_force, dt);
    
    return {next, achieved_force};
}

// ============================================================================
// Velocity-Based State Propagation
// ============================================================================

FullState DynamicsModel::stepKinematic(
    const FullState& state, 
    const BodyVelocity& cmd, 
    double dt) const
{
    // Clamp command
    BodyVelocity clamped_cmd = cmd;
    clamped_cmd.clamp(params_.limits.vx_max, params_.limits.vy_max, params_.limits.omega_max);
    
    // Direct velocity application (kinematic model)
    Pose2D new_pose = integratePose(state.pose(), clamped_cmd, dt);
    
    FullState next;
    next.setPose(new_pose);
    next.setVelocity(clamped_cmd);
    next.normalizeYaw();
    
    return next;
}

FullState DynamicsModel::stepWithSlip(
    const FullState& state,
    const BodyVelocity& cmd,
    double dt,
    double slip_factor) const
{
    // Apply slip model: v_y_effective = v_y - K_slip * v_x * omega
    // This models the lateral slip that occurs during turning
    BodyVelocity effective_cmd = cmd;
    double vy_slip = -slip_factor * cmd.vx * cmd.omega;
    effective_cmd.vy = cmd.vy + vy_slip;
    
    // Clamp
    effective_cmd.clamp(params_.limits.vx_max, params_.limits.vy_max, params_.limits.omega_max);
    
    // Integrate pose
    Pose2D new_pose = integratePose(state.pose(), effective_cmd, dt);
    
    FullState next;
    next.setPose(new_pose);
    next.setVelocity(effective_cmd);
    next.normalizeYaw();
    
    return next;
}

// ============================================================================
// Force Computation
// ============================================================================

BodyForce DynamicsModel::computeRequiredForce(
    const FullState& current,
    const FullState& desired_next,
    double dt) const
{
    if (dt < EPSILON) {
        return BodyForce{};
    }
    
    // Desired acceleration
    double ax = (desired_next.vx - current.vx) / dt;
    double ay = (desired_next.vy - current.vy) / dt;
    double alpha = (desired_next.omega - current.omega) / dt;
    
    // Remove inertial terms to get required applied force
    // a_applied = a_desired - a_inertial
    // a_inertial_x = vy * omega
    // a_inertial_y = -vx * omega
    double ax_applied = ax - current.vy * current.omega;
    double ay_applied = ay + current.vx * current.omega;
    
    // Force = mass * acceleration
    BodyForce force;
    force.Fx = params_.mass.mass * ax_applied;
    force.Fy = params_.mass.mass * ay_applied;
    force.Mz = params_.mass.Iz * alpha;
    
    return force;
}

BodyForce DynamicsModel::velocityToForce(
    const FullState& current,
    const BodyVelocity& cmd,
    double dt) const
{
    // Create desired next state
    FullState desired_next = current;
    desired_next.setVelocity(cmd);
    
    return computeRequiredForce(current, desired_next, dt);
}

// ============================================================================
// Wheel-Level Computations
// ============================================================================

std::array<BodyVelocity, NUM_WHEELS> DynamicsModel::computeWheelVelocities(
    const BodyVelocity& body_vel) const
{
    std::array<BodyVelocity, NUM_WHEELS> wheel_vels;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = params_.geometry.getWheelPosition(idx);
        wheel_vels[i] = computeWheelVelocity(body_vel, wheel_pos);
    }
    
    return wheel_vels;
}

std::array<double, NUM_WHEELS> DynamicsModel::computeWheelSlipAngles(
    const BodyVelocity& body_vel,
    const std::array<double, NUM_WHEELS>& steer_angles) const
{
    std::array<double, NUM_WHEELS> slip_angles;
    auto wheel_vels = computeWheelVelocities(body_vel);
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        auto [vx_tire, vy_tire] = transformToTireFrame(wheel_vels[i], steer_angles[i]);
        slip_angles[i] = TireModel::computeSlipAngle(vx_tire, vy_tire);
    }
    
    return slip_angles;
}

WheelForces DynamicsModel::computeWheelForces(
    const BodyVelocity& body_vel,
    const VehicleCommand8D& wheel_cmd) const
{
    WheelForces forces;
    auto wheel_vels = computeWheelVelocities(body_vel);
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        double steer = wheel_cmd.wheels[i].steer_angle;
        double wheel_speed = wheel_cmd.wheels[i].velocity / params_.geometry.tire_radius;
        
        // Transform wheel velocity to tire frame
        auto [vx_tire, vy_tire] = transformToTireFrame(wheel_vels[i], steer);
        
        // Compute slip
        double slip_angle = TireModel::computeSlipAngle(vx_tire, vy_tire);
        double slip_ratio = TireModel::computeSlipRatio(
            wheel_speed, vx_tire, params_.geometry.tire_radius);
        
        // Get normal load
        double Fz = params_.mass.getStaticWheelLoad(idx);
        
        // Compute tire force
        Force2D F_tire = tire_model_.computeCombinedForce(slip_angle, slip_ratio, Fz);
        
        // Transform back to body frame
        forces[i] = transformForceToBodyFrame(F_tire, steer);
    }
    
    return forces;
}

BodyForce DynamicsModel::wheelForcesToBodyForce(
    const WheelForces& wheel_forces,
    const std::array<double, NUM_WHEELS>& steer_angles) const
{
    BodyForce body_force;
    body_force.setZero();
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = params_.geometry.getWheelPosition(idx);
        
        // Force already in body frame
        body_force.Fx += wheel_forces[i].Fx;
        body_force.Fy += wheel_forces[i].Fy;
        
        // Moment: Mz = r × F = rx * Fy - ry * Fx
        body_force.Mz += wheel_pos.x * wheel_forces[i].Fy - wheel_pos.y * wheel_forces[i].Fx;
    }
    
    return body_force;
}

VehicleCommand8D DynamicsModel::velocityToWheelCommands(const BodyVelocity& body_vel) const
{
    VehicleCommand8D cmd;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = params_.geometry.getWheelPosition(idx);
        
        // Wheel velocity in body frame
        double vx_wheel = body_vel.vx - body_vel.omega * wheel_pos.y;
        double vy_wheel = body_vel.vy + body_vel.omega * wheel_pos.x;
        
        cmd.wheels[i].steer_angle = std::atan2(vy_wheel, vx_wheel);
        cmd.wheels[i].velocity = std::hypot(vx_wheel, vy_wheel);
    }
    
    return cmd;
}

// ============================================================================
// Private Helper Functions
// ============================================================================

BodyForce DynamicsModel::applyInertialCorrections(
    const BodyForce& input_force,
    const BodyVelocity& velocity) const
{
    // In body frame, we have pseudo-forces due to rotation
    // Coriolis acceleration:
    //   a_inertial_x = vy * omega  (must be subtracted from applied force effect)
    //   a_inertial_y = -vx * omega
    
    // The input force already includes what we want to achieve
    // We need to add the inertial terms as "free" accelerations
    // F_effective = F_applied (the input already accounts for this in stepForce)
    
    return input_force;  // No modification needed, handled in integration
}

BodyVelocity DynamicsModel::integrateAcceleration(
    const BodyVelocity& current_vel,
    const BodyForce& force,
    double dt) const
{
    // Compute acceleration from force
    // In body frame with Coriolis terms:
    // dvx/dt = Fx/m + vy*omega
    // dvy/dt = Fy/m - vx*omega
    // domega/dt = Mz/Iz
    
    double ax = force.Fx / params_.mass.mass + current_vel.vy * current_vel.omega;
    double ay = force.Fy / params_.mass.mass - current_vel.vx * current_vel.omega;
    double alpha = force.Mz / params_.mass.Iz;
    
    // Simple Euler integration
    BodyVelocity new_vel;
    new_vel.vx = current_vel.vx + ax * dt;
    new_vel.vy = current_vel.vy + ay * dt;
    new_vel.omega = current_vel.omega + alpha * dt;
    
    return new_vel;
}

Pose2D DynamicsModel::integratePose(
    const Pose2D& current_pose,
    const BodyVelocity& velocity,
    double dt) const
{
    // Transform body velocity to world frame and integrate
    double cos_yaw = std::cos(current_pose.yaw);
    double sin_yaw = std::sin(current_pose.yaw);
    
    // World frame velocities
    double vx_world = velocity.vx * cos_yaw - velocity.vy * sin_yaw;
    double vy_world = velocity.vx * sin_yaw + velocity.vy * cos_yaw;
    
    // Integrate position
    Pose2D new_pose;
    new_pose.x = current_pose.x + vx_world * dt;
    new_pose.y = current_pose.y + vy_world * dt;
    new_pose.yaw = current_pose.yaw + velocity.omega * dt;
    new_pose.normalizeYaw();
    
    return new_pose;
}

} // namespace mppi_tf
