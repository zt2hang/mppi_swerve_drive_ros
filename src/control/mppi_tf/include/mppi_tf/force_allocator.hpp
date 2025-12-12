#pragma once

/**
 * @file force_allocator.hpp
 * @brief Optimal force allocation for over-actuated swerve drive
 * 
 * This module solves the control allocation problem for a 4-wheel swerve drive:
 * Given desired body forces (Fx, Fy, Mz), find individual wheel forces that:
 * 1. Sum to the desired body force/moment
 * 2. Respect friction circle constraints on each wheel
 * 3. Minimize total force usage (efficiency)
 * 4. Maximize friction margin (safety)
 * 
 * The allocation is formulated as a constrained optimization:
 *   min  Σ||Fi||² + w_margin * Σ(1 - margin_i)²
 *   s.t. Σ Fi = F_des
 *        Σ ri × Fi = Mz_des
 *        ||Fi|| ≤ μ * Fz_i  (friction circle)
 * 
 * Solution Methods:
 * 1. Pseudo-inverse with iterative saturation handling
 * 2. Weighted least squares with friction margin objective
 * 
 * References:
 * - Johansen & Fossen "Control Allocation - A Survey" (2013)
 * - de Novellis et al. "Optimal wheel torque distribution..." (2014)
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include "mppi_tf/tire_model.hpp"
#include <Eigen/Dense>
#include <array>

namespace mppi_tf
{

/**
 * @brief Optimal force allocation for 4-wheel swerve drive
 */
class ForceAllocator
{
public:
    /**
     * @brief Construct allocator with vehicle parameters
     * @param geometry Vehicle geometry (wheel positions)
     * @param mass Vehicle mass properties
     * @param alloc_params Allocation tuning parameters
     */
    ForceAllocator(
        const VehicleGeometry& geometry,
        const VehicleMass& mass,
        const AllocationParams& params
    );

    /**
     * @brief Update allocation parameters
     */
    void setParams(const AllocationParams& params) { params_ = params; }

    /**
     * @brief Set current friction coefficient (from estimator)
     */
    void setFrictionCoeff(double mu) { mu_ = std::clamp(mu, 0.1, 1.5); }

    /**
     * @brief Set normal loads (can vary with load transfer)
     */
    void setNormalLoads(const std::array<double, NUM_WHEELS>& Fz) { Fz_ = Fz; }

    /**
     * @brief Set steering angles (determines force direction capabilities)
     */
    void setSteeringAngles(const std::array<double, NUM_WHEELS>& steer) { 
        steer_angles_ = steer; 
    }

    // ========================================================================
    // Main Allocation Methods
    // ========================================================================

    /**
     * @brief Allocate body force to wheel forces (main interface)
     * 
     * This is the primary allocation method. It finds optimal wheel forces
     * that achieve the desired body force while respecting friction limits.
     * 
     * @param desired Body force/moment demand
     * @return Allocation result with wheel forces and diagnostics
     */
    AllocationResult allocate(const BodyForce& desired);

    /**
     * @brief Quick allocation using pseudo-inverse (faster, less optimal)
     * 
     * Uses a simple pseudo-inverse followed by saturation. Suitable for
     * real-time applications where the full optimization is too slow.
     * 
     * @param desired Body force/moment demand
     * @return Allocated wheel forces
     */
    WheelForces allocatePseudoInverse(const BodyForce& desired);

    /**
     * @brief Allocate with explicit friction margin target
     * 
     * Attempts to maintain a specified margin from friction limit on all wheels.
     * Useful for safety-critical applications.
     * 
     * @param desired Body force/moment demand
     * @param margin_target Target friction margin (0 to 1)
     * @return Allocation result
     */
    AllocationResult allocateWithMargin(const BodyForce& desired, double margin_target);

    // ========================================================================
    // Analysis Methods
    // ========================================================================

    /**
     * @brief Check if desired force is achievable
     * @param desired Body force/moment demand
     * @return true if force can be achieved within friction limits
     */
    bool isFeasible(const BodyForce& desired) const;

    /**
     * @brief Get maximum achievable force in a given direction
     * @param direction Unit vector in body frame (Fx, Fy normalized)
     * @return Maximum force magnitude achievable
     */
    double getMaxForceInDirection(const Eigen::Vector2d& direction) const;

    /**
     * @brief Get maximum achievable yaw moment
     * @return Maximum Mz magnitude [Nm]
     */
    double getMaxYawMoment() const;

    /**
     * @brief Compute total friction utilization for given wheel forces
     * @param forces Wheel forces
     * @return Average friction utilization [0, 1]
     */
    double computeTotalUtilization(const WheelForces& forces) const;

    /**
     * @brief Compute minimum friction margin
     * @param forces Wheel forces  
     * @return Minimum margin (1 - utilization) across all wheels
     */
    double computeMinMargin(const WheelForces& forces) const;

    // ========================================================================
    // Conversion Utilities
    // ========================================================================

    /**
     * @brief Convert wheel forces to wheel commands
     * 
     * Given desired wheel forces in body frame, computes the required
     * wheel velocities and steering angles.
     * 
     * @param wheel_forces Desired forces at each wheel (body frame)
     * @param body_vel Current body velocity (for slip calculation)
     * @param tire_model Tire model for inverse force computation
     * @return Wheel commands (velocity, steering angle)
     */
    VehicleCommand8D forcesToCommands(
        const WheelForces& wheel_forces,
        const BodyVelocity& body_vel,
        const TireModel& tire_model
    ) const;

    /**
     * @brief Convert body velocity to wheel commands (kinematic)
     * 
     * Pure kinematic conversion without force consideration.
     * 
     * @param body_vel Desired body velocity
     * @return Wheel commands
     */
    VehicleCommand8D velocityToCommands(const BodyVelocity& body_vel) const;

    /**
     * @brief Compute body force from wheel forces
     * @param wheel_forces Forces at each wheel
     * @return Resultant body force
     */
    BodyForce computeBodyForce(const WheelForces& wheel_forces) const;

private:
    VehicleGeometry geometry_;
    VehicleMass mass_;
    AllocationParams params_;
    
    double mu_ = 0.9;  // Current friction coefficient
    std::array<double, NUM_WHEELS> Fz_;  // Normal loads
    std::array<double, NUM_WHEELS> steer_angles_;  // Steering angles
    
    // Allocation matrix (maps wheel forces to body forces)
    // B: [3 x 8] matrix where body_force = B * wheel_forces
    Eigen::Matrix<double, 3, 8> B_;
    
    // Pseudo-inverse of allocation matrix
    Eigen::Matrix<double, 8, 3> B_pinv_;

    /**
     * @brief Update allocation matrix based on current steering angles
     */
    void updateAllocationMatrix();

    /**
     * @brief Iterative saturation handling
     * 
     * When some wheels saturate, this redistributes their excess
     * force to unsaturated wheels.
     * 
     * @param initial_forces Initial force allocation (may exceed limits)
     * @param desired Desired body force
     * @return Saturated forces and achieved body force
     */
    std::pair<WheelForces, BodyForce> handleSaturation(
        const WheelForces& initial_forces,
        const BodyForce& desired
    );

    /**
     * @brief Single iteration of weighted least squares allocation
     */
    WheelForces weightedLeastSquaresStep(
        const BodyForce& desired,
        const Eigen::Matrix<double, 8, 1>& weights
    );

    /**
     * @brief Compute weights based on friction margin
     */
    Eigen::Matrix<double, 8, 1> computeMarginWeights(const WheelForces& current) const;
};

} // namespace mppi_tf
