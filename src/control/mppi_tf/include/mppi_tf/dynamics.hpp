#pragma once

/**
 * @file dynamics.hpp
 * @brief Force-based dynamics model for omnidirectional swerve drive
 * 
 * This module implements the vehicle dynamics equations that propagate
 * state given applied forces. Unlike kinematic models that directly
 * apply velocity commands, this model considers:
 * 
 * 1. Force-to-acceleration dynamics (Newton's laws)
 * 2. Tire force generation with slip effects
 * 3. Inertial forces during rotation (centripetal acceleration)
 * 4. Optional load transfer effects
 * 
 * State: (x, y, θ, vx, vy, ω) - position, orientation, velocities
 * Input: (Fx, Fy, Mz) - body forces and moment
 * 
 * Dynamics Equations (body frame):
 *   ẍ_body = Fx/m + vy*ω   (includes centripetal term)
 *   ÿ_body = Fy/m - vx*ω
 *   ω̈ = Mz/Iz
 * 
 * The model supports multiple fidelity levels:
 * - Kinematic: Direct velocity control (instant force)
 * - Force-based: Acceleration from forces
 * - Tire-force: Full tire slip model with force generation
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include "mppi_tf/tire_model.hpp"
#include "mppi_tf/force_allocator.hpp"

namespace mppi_tf
{

/**
 * @brief Dynamics model fidelity level
 */
enum class DynamicsMode
{
    KINEMATIC,      // Direct velocity control (no dynamics)
    FORCE_BASED,    // Force-to-acceleration dynamics
    FULL_TIRE       // Complete tire model with slip
};

/**
 * @brief Force-based dynamics model for swerve drive
 */
class DynamicsModel
{
public:
    /**
     * @brief Construct dynamics model
     * @param vehicle Vehicle parameters
     * @param mode Dynamics fidelity level
     */
    DynamicsModel(const VehicleParams& vehicle, DynamicsMode mode = DynamicsMode::FORCE_BASED);

    /**
     * @brief Set dynamics mode
     */
    void setMode(DynamicsMode mode) { mode_ = mode; }
    DynamicsMode getMode() const { return mode_; }

    /**
     * @brief Update tire model parameters (from estimator)
     */
    void setTireParams(double C_alpha, double mu);

    // ========================================================================
    // State Propagation - Force Input
    // ========================================================================

    /**
     * @brief Propagate state given body force input
     * 
     * This is the primary interface for force-space MPPI.
     * 
     * @param state Current state (x, y, θ, vx, vy, ω)
     * @param force Body force/moment input (Fx, Fy, Mz)
     * @param dt Time step [s]
     * @return Next state
     */
    FullState stepForce(const FullState& state, const BodyForce& force, double dt) const;

    /**
     * @brief Propagate state with full tire model
     * 
     * Given desired body force, allocates to wheels, computes actual
     * tire forces considering slip, and propagates dynamics.
     * 
     * @param state Current state
     * @param desired_force Desired body force
     * @param dt Time step
     * @return Next state and achieved force
     */
    std::pair<FullState, BodyForce> stepTireModel(
        const FullState& state, 
        const BodyForce& desired_force, 
        double dt) const;

    // ========================================================================
    // State Propagation - Velocity Input
    // ========================================================================

    /**
     * @brief Propagate state given velocity command (kinematic)
     * 
     * For compatibility with velocity-space sampling.
     * 
     * @param state Current state
     * @param cmd Velocity command
     * @param dt Time step
     * @return Next state
     */
    FullState stepKinematic(const FullState& state, const BodyVelocity& cmd, double dt) const;

    /**
     * @brief Propagate state with velocity input and slip model
     * 
     * Similar to MPPI-HC: applies slip correction to velocity.
     * 
     * @param state Current state
     * @param cmd Velocity command  
     * @param dt Time step
     * @param slip_factor Estimated slip factor
     * @return Next state
     */
    FullState stepWithSlip(
        const FullState& state, 
        const BodyVelocity& cmd, 
        double dt,
        double slip_factor) const;

    // ========================================================================
    // Force Computation
    // ========================================================================

    /**
     * @brief Compute required force for desired acceleration
     * 
     * Inverse dynamics: given desired next state, compute needed force.
     * 
     * @param current Current state
     * @param desired_next Desired next state
     * @param dt Time step
     * @return Required body force
     */
    BodyForce computeRequiredForce(
        const FullState& current,
        const FullState& desired_next,
        double dt) const;

    /**
     * @brief Compute force from velocity command
     * 
     * Computes the force needed to achieve a velocity change.
     * 
     * @param current Current state
     * @param cmd Desired velocity
     * @param dt Time step
     * @return Required body force
     */
    BodyForce velocityToForce(
        const FullState& current,
        const BodyVelocity& cmd,
        double dt) const;

    // ========================================================================
    // Wheel-Level Computations
    // ========================================================================

    /**
     * @brief Compute wheel velocities from body velocity
     */
    std::array<BodyVelocity, NUM_WHEELS> computeWheelVelocities(
        const BodyVelocity& body_vel) const;

    /**
     * @brief Compute wheel slip angles
     */
    std::array<double, NUM_WHEELS> computeWheelSlipAngles(
        const BodyVelocity& body_vel,
        const std::array<double, NUM_WHEELS>& steer_angles) const;

    /**
     * @brief Compute wheel forces from body velocity and desired wheel commands
     */
    WheelForces computeWheelForces(
        const BodyVelocity& body_vel,
        const VehicleCommand8D& wheel_cmd) const;

    /**
     * @brief Convert wheel forces to body force
     */
    BodyForce wheelForcesToBodyForce(
        const WheelForces& wheel_forces,
        const std::array<double, NUM_WHEELS>& steer_angles) const;

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * @brief Convert body velocity to wheel commands (kinematic)
     */
    VehicleCommand8D velocityToWheelCommands(const BodyVelocity& body_vel) const;

    /**
     * @brief Get force limits based on vehicle parameters
     */
    BodyForce getForceLimits() const;

    /**
     * @brief Check if force is within limits
     */
    bool isForceValid(const BodyForce& force) const;

    /**
     * @brief Clamp force to limits
     */
    BodyForce clampForce(const BodyForce& force) const;

    const VehicleParams& getParams() const { return params_; }

private:
    VehicleParams params_;
    DynamicsMode mode_;
    
    // Tire model for FULL_TIRE mode
    mutable TireModel tire_model_;
    mutable ForceAllocator force_allocator_;
    
    // Current tire parameters (from estimator)
    double C_alpha_current_;
    double mu_current_;

    /**
     * @brief Apply inertial force corrections (Coriolis terms)
     */
    BodyForce applyInertialCorrections(
        const BodyForce& input_force,
        const BodyVelocity& velocity) const;

    /**
     * @brief Integrate acceleration to get new velocity
     */
    BodyVelocity integrateAcceleration(
        const BodyVelocity& current_vel,
        const BodyForce& force,
        double dt) const;

    /**
     * @brief Integrate velocity to get new pose
     */
    Pose2D integratePose(
        const Pose2D& current_pose,
        const BodyVelocity& velocity,
        double dt) const;
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline BodyForce DynamicsModel::getForceLimits() const
{
    return BodyForce{
        params_.getMaxLongForce(),
        params_.getMaxLatForce(),
        params_.getMaxYawMoment()
    };
}

inline bool DynamicsModel::isForceValid(const BodyForce& force) const
{
    auto limits = getForceLimits();
    return std::abs(force.Fx) <= limits.Fx * 1.1 &&
           std::abs(force.Fy) <= limits.Fy * 1.1 &&
           std::abs(force.Mz) <= limits.Mz * 1.1;
}

inline BodyForce DynamicsModel::clampForce(const BodyForce& force) const
{
    auto limits = getForceLimits();
    return BodyForce{
        std::clamp(force.Fx, -limits.Fx, limits.Fx),
        std::clamp(force.Fy, -limits.Fy, limits.Fy),
        std::clamp(force.Mz, -limits.Mz, limits.Mz)
    };
}

} // namespace mppi_tf
