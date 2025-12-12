/**
 * @file force_allocator.cpp
 * @brief Implementation of optimal force allocation
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/force_allocator.hpp"
#include <iostream>

namespace mppi_tf
{

ForceAllocator::ForceAllocator(
    const VehicleGeometry& geometry,
    const VehicleMass& mass,
    const AllocationParams& params)
    : geometry_(geometry)
    , mass_(mass)
    , params_(params)
{
    // Initialize normal loads (static distribution)
    for (int i = 0; i < NUM_WHEELS; ++i) {
        Fz_[i] = mass_.getStaticWheelLoad(static_cast<WheelIndex>(i));
        steer_angles_[i] = 0.0;
    }
    
    updateAllocationMatrix();
}

void ForceAllocator::updateAllocationMatrix()
{
    // The allocation matrix B maps wheel forces (in body frame) to body forces
    // Body force = B * [Fx_fl, Fy_fl, Fx_fr, Fy_fr, Fx_rl, Fy_rl, Fx_rr, Fy_rr]^T
    //
    // For each wheel i at position (rx_i, ry_i):
    //   Contribution to Fx_body = Fx_i
    //   Contribution to Fy_body = Fy_i
    //   Contribution to Mz_body = rx_i * Fy_i - ry_i * Fx_i
    
    B_.setZero();
    
    // Get wheel positions
    std::array<Position2D, NUM_WHEELS> wheel_pos;
    wheel_pos[FRONT_LEFT] = geometry_.getWheelPosition(FRONT_LEFT);
    wheel_pos[FRONT_RIGHT] = geometry_.getWheelPosition(FRONT_RIGHT);
    wheel_pos[REAR_LEFT] = geometry_.getWheelPosition(REAR_LEFT);
    wheel_pos[REAR_RIGHT] = geometry_.getWheelPosition(REAR_RIGHT);
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        int col_fx = 2 * i;      // Column for Fx of wheel i
        int col_fy = 2 * i + 1;  // Column for Fy of wheel i
        
        double rx = wheel_pos[i].x;
        double ry = wheel_pos[i].y;
        
        // Fx_body contribution
        B_(0, col_fx) = 1.0;
        B_(0, col_fy) = 0.0;
        
        // Fy_body contribution
        B_(1, col_fx) = 0.0;
        B_(1, col_fy) = 1.0;
        
        // Mz_body contribution: Mz = r × F = rx * Fy - ry * Fx
        B_(2, col_fx) = -ry;
        B_(2, col_fy) = rx;
    }
    
    // Compute pseudo-inverse: B_pinv = B^T * (B * B^T)^{-1}
    Eigen::Matrix3d BBt = B_ * B_.transpose();
    
    // Add regularization for numerical stability
    BBt += Eigen::Matrix3d::Identity() * 1e-6;
    
    B_pinv_ = B_.transpose() * BBt.inverse();
}

// ============================================================================
// Main Allocation Methods
// ============================================================================

AllocationResult ForceAllocator::allocate(const BodyForce& desired)
{
    AllocationResult result;
    result.iterations = 0;
    
    // Start with pseudo-inverse solution
    Eigen::Vector3d f_des;
    f_des << desired.Fx, desired.Fy, desired.Mz;
    
    Eigen::Matrix<double, 8, 1> f_wheels = B_pinv_ * f_des;
    
    // Convert to WheelForces structure
    WheelForces initial;
    for (int i = 0; i < NUM_WHEELS; ++i) {
        initial[i].Fx = f_wheels(2 * i);
        initial[i].Fy = f_wheels(2 * i + 1);
    }
    
    // Handle saturation iteratively
    auto [saturated, achieved] = handleSaturation(initial, desired);
    
    result.wheel_forces = saturated;
    result.achieved_force = achieved;
    result.total_utilization = computeTotalUtilization(saturated);
    result.min_margin = computeMinMargin(saturated);
    result.is_saturated = result.min_margin < 0.01;
    
    return result;
}

WheelForces ForceAllocator::allocatePseudoInverse(const BodyForce& desired)
{
    Eigen::Vector3d f_des;
    f_des << desired.Fx, desired.Fy, desired.Mz;
    
    Eigen::Matrix<double, 8, 1> f_wheels = B_pinv_ * f_des;
    
    WheelForces forces;
    for (int i = 0; i < NUM_WHEELS; ++i) {
        forces[i].Fx = f_wheels(2 * i);
        forces[i].Fy = f_wheels(2 * i + 1);
        
        // Saturate to friction circle
        double F_max = mu_ * Fz_[i];
        forces[i].saturate(F_max);
    }
    
    return forces;
}

AllocationResult ForceAllocator::allocateWithMargin(const BodyForce& desired, double margin_target)
{
    AllocationResult result;
    
    // Scale down desired force to ensure margin
    double scale = 1.0 - margin_target;
    BodyForce scaled_desired = desired * scale;
    
    // Allocate scaled force
    result = allocate(scaled_desired);
    
    // If we have margin left, try to get closer to original desired
    if (result.min_margin > margin_target) {
        // Binary search for optimal scale
        double lo = scale;
        double hi = 1.0;
        
        for (int iter = 0; iter < 5; ++iter) {
            double mid = (lo + hi) / 2.0;
            BodyForce test_force = desired * mid;
            auto test_result = allocate(test_force);
            
            if (test_result.min_margin >= margin_target) {
                lo = mid;
                result = test_result;
            } else {
                hi = mid;
            }
        }
    }
    
    return result;
}

// ============================================================================
// Saturation Handling
// ============================================================================

std::pair<WheelForces, BodyForce> ForceAllocator::handleSaturation(
    const WheelForces& initial_forces,
    const BodyForce& desired)
{
    WheelForces current = initial_forces;
    
    // Check which wheels are saturated and by how much
    std::array<bool, NUM_WHEELS> saturated;
    std::array<double, NUM_WHEELS> saturation_factor;
    
    bool any_saturated = false;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double F_max = mu_ * Fz_[i];
        double F_mag = current[i].magnitude();
        
        if (F_mag > F_max && F_mag > EPSILON) {
            saturated[i] = true;
            saturation_factor[i] = F_max / F_mag;
            any_saturated = true;
        } else {
            saturated[i] = false;
            saturation_factor[i] = 1.0;
        }
    }
    
    if (!any_saturated) {
        // No saturation, return as is
        return {current, computeBodyForce(current)};
    }
    
    // Iterative redistribution
    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        // Saturate wheels that exceed limits
        for (int i = 0; i < NUM_WHEELS; ++i) {
            if (saturated[i]) {
                double F_max = mu_ * Fz_[i];
                current[i].saturate(F_max);
            }
        }
        
        // Compute achieved force
        BodyForce achieved = computeBodyForce(current);
        
        // Compute error
        Eigen::Vector3d error;
        error << desired.Fx - achieved.Fx, 
                 desired.Fy - achieved.Fy, 
                 desired.Mz - achieved.Mz;
        
        if (error.norm() < params_.tolerance) {
            break;
        }
        
        // Build reduced allocation matrix (excluding saturated wheels)
        // This redistributes the error to unsaturated wheels
        std::vector<int> free_indices;
        for (int i = 0; i < NUM_WHEELS; ++i) {
            if (!saturated[i]) {
                free_indices.push_back(2 * i);
                free_indices.push_back(2 * i + 1);
            }
        }
        
        if (free_indices.empty()) {
            // All wheels saturated, can't do better
            break;
        }
        
        // Extract columns for free wheels
        int n_free = free_indices.size();
        Eigen::MatrixXd B_free(3, n_free);
        for (int j = 0; j < n_free; ++j) {
            B_free.col(j) = B_.col(free_indices[j]);
        }
        
        // Compute correction via pseudo-inverse
        Eigen::MatrixXd B_free_pinv = B_free.transpose() * 
            (B_free * B_free.transpose() + Eigen::Matrix3d::Identity() * 1e-6).inverse();
        
        Eigen::VectorXd delta_f = B_free_pinv * error;
        
        // Apply correction to free wheels
        for (int j = 0; j < n_free; ++j) {
            int wheel_idx = free_indices[j] / 2;
            bool is_fx = (free_indices[j] % 2 == 0);
            
            if (is_fx) {
                current[wheel_idx].Fx += delta_f(j);
            } else {
                current[wheel_idx].Fy += delta_f(j);
            }
        }
        
        // Re-check saturation
        any_saturated = false;
        for (int i = 0; i < NUM_WHEELS; ++i) {
            double F_max = mu_ * Fz_[i];
            double F_mag = current[i].magnitude();
            
            if (F_mag > F_max * 1.01) {  // 1% tolerance
                saturated[i] = true;
                any_saturated = true;
            }
        }
        
        if (!any_saturated) {
            break;
        }
    }
    
    // Final saturation pass
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double F_max = mu_ * Fz_[i];
        current[i].saturate(F_max);
    }
    
    return {current, computeBodyForce(current)};
}

// ============================================================================
// Analysis Methods
// ============================================================================

bool ForceAllocator::isFeasible(const BodyForce& desired) const
{
    // Quick feasibility check using pseudo-inverse
    Eigen::Vector3d f_des;
    f_des << desired.Fx, desired.Fy, desired.Mz;
    
    Eigen::Matrix<double, 8, 1> f_wheels = B_pinv_ * f_des;
    
    // Check if any wheel exceeds its friction limit
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double Fx = f_wheels(2 * i);
        double Fy = f_wheels(2 * i + 1);
        double F_max = mu_ * Fz_[i];
        
        if (Fx * Fx + Fy * Fy > F_max * F_max) {
            // This doesn't mean infeasible (redistribution might help)
            // but indicates potential issues
            return false;
        }
    }
    
    return true;
}

double ForceAllocator::getMaxForceInDirection(const Eigen::Vector2d& direction) const
{
    // Binary search for maximum achievable force in direction
    Eigen::Vector2d dir = direction.normalized();
    
    double lo = 0.0;
    double hi = 4.0 * mu_ * mass_.mass * GRAVITY;  // Upper bound
    
    for (int i = 0; i < 20; ++i) {
        double mid = (lo + hi) / 2.0;
        BodyForce test;
        test.Fx = mid * dir(0);
        test.Fy = mid * dir(1);
        test.Mz = 0.0;
        
        if (isFeasible(test)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    
    return lo;
}

double ForceAllocator::getMaxYawMoment() const
{
    // Maximum yaw moment when all wheels contribute maximally
    double Mz_max = 0.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        Position2D pos = geometry_.getWheelPosition(static_cast<WheelIndex>(i));
        double r = std::hypot(pos.x, pos.y);
        double F_max = mu_ * Fz_[i];
        Mz_max += r * F_max;
    }
    
    return Mz_max;
}

double ForceAllocator::computeTotalUtilization(const WheelForces& forces) const
{
    double total = 0.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double F_max = mu_ * Fz_[i];
        double F_mag = forces[i].magnitude();
        total += (F_max > EPSILON) ? F_mag / F_max : 0.0;
    }
    
    return total / NUM_WHEELS;
}

double ForceAllocator::computeMinMargin(const WheelForces& forces) const
{
    double min_margin = 1.0;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double F_max = mu_ * Fz_[i];
        double F_mag = forces[i].magnitude();
        double utilization = (F_max > EPSILON) ? F_mag / F_max : 0.0;
        double margin = 1.0 - utilization;
        min_margin = std::min(min_margin, margin);
    }
    
    return min_margin;
}

// ============================================================================
// Conversion Utilities
// ============================================================================

BodyForce ForceAllocator::computeBodyForce(const WheelForces& wheel_forces) const
{
    Eigen::Matrix<double, 8, 1> f_vec;
    for (int i = 0; i < NUM_WHEELS; ++i) {
        f_vec(2 * i) = wheel_forces[i].Fx;
        f_vec(2 * i + 1) = wheel_forces[i].Fy;
    }
    
    Eigen::Vector3d body = B_ * f_vec;
    
    return BodyForce{body(0), body(1), body(2)};
}

VehicleCommand8D ForceAllocator::forcesToCommands(
    const WheelForces& wheel_forces,
    const BodyVelocity& body_vel,
    const TireModel& tire_model) const
{
    VehicleCommand8D cmd;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = geometry_.getWheelPosition(idx);
        
        // Get wheel velocity in body frame
        BodyVelocity wheel_vel = computeWheelVelocity(body_vel, wheel_pos);
        
        // Desired force in body frame
        const Force2D& F_body = wheel_forces[i];
        
        // Steering angle is the direction of the desired force
        // (assuming we want to point the wheel in the force direction)
        double steer = std::atan2(F_body.Fy, F_body.Fx);
        
        // Alternative: align with velocity + force correction
        double vel_heading = std::atan2(wheel_vel.vy, wheel_vel.vx);
        double force_heading = std::atan2(F_body.Fy, F_body.Fx);
        
        // Blend based on speed (at low speed, use force direction)
        double speed = wheel_vel.speed();
        double blend = std::min(1.0, speed / 0.5);  // Full velocity alignment above 0.5 m/s
        
        if (speed > 0.1) {
            // At higher speeds, limit deviation from velocity direction
            double delta = std::remainder(force_heading - vel_heading, 2 * M_PI);
            delta = std::clamp(delta, -M_PI / 4, M_PI / 4);  // Max 45 deg deviation
            steer = vel_heading + delta;
        }
        
        // Transform desired force to tire frame
        Force2D F_tire;
        double cos_d = std::cos(steer);
        double sin_d = std::sin(steer);
        F_tire.Fx = F_body.Fx * cos_d + F_body.Fy * sin_d;
        F_tire.Fy = -F_body.Fx * sin_d + F_body.Fy * cos_d;
        
        // Use tire model inverse to get required slip
        double slip_angle = tire_model.computeRequiredSlipAngle(F_tire.Fy, Fz_[i]);
        
        // Wheel velocity magnitude (from body velocity at wheel)
        auto [vx_tire, vy_tire] = transformToTireFrame(wheel_vel, steer);
        
        // Desired wheel velocity accounting for slip angle
        // For small slip angles: vy_tire ≈ vx_tire * tan(alpha)
        // So we need: wheel_speed ≈ vx_tire / cos(slip_angle)
        double wheel_speed = std::hypot(vx_tire, vy_tire);
        
        cmd.wheels[i].steer_angle = steer;
        cmd.wheels[i].velocity = wheel_speed;
    }
    
    return cmd;
}

VehicleCommand8D ForceAllocator::velocityToCommands(const BodyVelocity& body_vel) const
{
    VehicleCommand8D cmd;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        WheelIndex idx = static_cast<WheelIndex>(i);
        Position2D wheel_pos = geometry_.getWheelPosition(idx);
        
        // Compute wheel velocity in body frame
        double vx_wheel = body_vel.vx - body_vel.omega * wheel_pos.y;
        double vy_wheel = body_vel.vy + body_vel.omega * wheel_pos.x;
        
        // Steering angle and wheel speed
        cmd.wheels[i].steer_angle = std::atan2(vy_wheel, vx_wheel);
        cmd.wheels[i].velocity = std::hypot(vx_wheel, vy_wheel);
    }
    
    return cmd;
}

// ============================================================================
// Weighted Least Squares (for margin-aware allocation)
// ============================================================================

WheelForces ForceAllocator::weightedLeastSquaresStep(
    const BodyForce& desired,
    const Eigen::Matrix<double, 8, 1>& weights)
{
    // Weighted pseudo-inverse: B_pinv_w = W^{-1} * B^T * (B * W^{-1} * B^T)^{-1}
    Eigen::Matrix<double, 8, 8> W = weights.asDiagonal();
    Eigen::Matrix<double, 8, 8> W_inv = W.inverse();
    
    Eigen::Matrix3d BWBt = B_ * W_inv * B_.transpose();
    BWBt += Eigen::Matrix3d::Identity() * 1e-6;  // Regularization
    
    Eigen::Matrix<double, 8, 3> B_pinv_w = W_inv * B_.transpose() * BWBt.inverse();
    
    Eigen::Vector3d f_des;
    f_des << desired.Fx, desired.Fy, desired.Mz;
    
    Eigen::Matrix<double, 8, 1> f_wheels = B_pinv_w * f_des;
    
    WheelForces forces;
    for (int i = 0; i < NUM_WHEELS; ++i) {
        forces[i].Fx = f_wheels(2 * i);
        forces[i].Fy = f_wheels(2 * i + 1);
    }
    
    return forces;
}

Eigen::Matrix<double, 8, 1> ForceAllocator::computeMarginWeights(
    const WheelForces& current) const
{
    Eigen::Matrix<double, 8, 1> weights;
    
    for (int i = 0; i < NUM_WHEELS; ++i) {
        double F_max = mu_ * Fz_[i];
        double F_mag = current[i].magnitude();
        double utilization = (F_max > EPSILON) ? F_mag / F_max : 0.0;
        
        // Higher weight = less force allocated to this wheel
        // Wheels near saturation get higher weights
        double margin = 1.0 - utilization;
        double weight = 1.0 / (margin * margin + 0.01);  // Avoid division by zero
        
        weights(2 * i) = weight;
        weights(2 * i + 1) = weight;
    }
    
    return weights;
}

} // namespace mppi_tf
