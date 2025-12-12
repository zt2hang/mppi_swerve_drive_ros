#pragma once

/**
 * @file tire_model.hpp
 * @brief Physics-based tire force model using Pacejka Magic Formula
 * 
 * This module implements a comprehensive tire model that captures:
 * 1. Lateral force generation via slip angle (cornering)
 * 2. Longitudinal force generation via slip ratio (traction/braking)
 * 3. Combined slip behavior using friction ellipse
 * 4. Load-dependent force saturation
 * 
 * The model is based on the Pacejka "Magic Formula" which provides
 * excellent empirical fit to tire data:
 *   F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
 * 
 * Key Features:
 * - Nonlinear force-slip characteristics
 * - Friction circle constraint
 * - Load transfer effects
 * - Combined slip coupling
 * 
 * References:
 * - Pacejka, H.B. "Tire and Vehicle Dynamics" (3rd ed., 2012)
 * - Milliken & Milliken "Race Car Vehicle Dynamics" (1995)
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/types.hpp"
#include <cmath>
#include <algorithm>

namespace mppi_tf
{

/**
 * @brief Tire force model based on Pacejka Magic Formula
 * 
 * This class computes tire forces given slip conditions and normal load,
 * handling both pure slip (lateral-only or longitudinal-only) and combined
 * slip scenarios.
 */
class TireModel
{
public:
    /**
     * @brief Construct tire model with given parameters
     * @param params Tire parameters (stiffness, friction coefficients, etc.)
     */
    explicit TireModel(const TireParams& params);

    /**
     * @brief Update tire parameters (for online adaptation)
     * @param params New tire parameters
     */
    void setParams(const TireParams& params) { params_ = params; }
    const TireParams& getParams() const { return params_; }

    /**
     * @brief Set estimated friction coefficient
     * @param mu Estimated road friction coefficient
     */
    void setFrictionCoeff(double mu) { mu_current_ = std::clamp(mu, 0.1, 1.5); }
    double getFrictionCoeff() const { return mu_current_; }

    /**
     * @brief Set estimated cornering stiffness
     * @param C_alpha Estimated cornering stiffness [N/rad]
     */
    void setCorneringStiffness(double C_alpha) { 
        C_alpha_current_ = std::clamp(C_alpha, 10000.0, 200000.0); 
    }
    double getCorneringStiffness() const { return C_alpha_current_; }

    // ========================================================================
    // Pure Slip Force Computation
    // ========================================================================

    /**
     * @brief Compute pure lateral force (cornering force)
     * 
     * Uses the Pacejka Magic Formula:
     *   Fy = D * sin(C * atan(B*α - E*(B*α - atan(B*α))))
     * 
     * where:
     *   D = μ * Fz (peak force)
     *   B = C_α / (C * D) (stiffness factor)
     *   C ≈ 1.9 (shape factor for lateral)
     *   E ≈ 0.97 (curvature factor)
     * 
     * @param slip_angle Tire slip angle [rad]
     * @param Fz Normal load [N]
     * @return Lateral force [N] (positive = force in positive y direction)
     */
    double computeLateralForce(double slip_angle, double Fz) const;

    /**
     * @brief Compute pure longitudinal force (traction/braking)
     * 
     * Uses similar Magic Formula structure:
     *   Fx = D * sin(C * atan(B*κ - E*(B*κ - atan(B*κ))))
     * 
     * @param slip_ratio Longitudinal slip ratio [-]
     * @param Fz Normal load [N]
     * @return Longitudinal force [N] (positive = driving force)
     */
    double computeLongitudinalForce(double slip_ratio, double Fz) const;

    // ========================================================================
    // Combined Slip Force Computation
    // ========================================================================

    /**
     * @brief Compute forces under combined slip conditions
     * 
     * When both slip angle and slip ratio are present, the forces
     * interact through the friction ellipse. The combined forces
     * are computed using:
     * 
     *   σ = sqrt((κ/κ_peak)² + (tan(α)/tan(α_peak))²)
     *   Fx_comb = Fx0 * κ/(κ_peak * σ)
     *   Fy_comb = Fy0 * tan(α)/(tan(α_peak) * σ)
     * 
     * This ensures the resultant force stays within the friction circle.
     * 
     * @param slip_angle Slip angle [rad]
     * @param slip_ratio Slip ratio [-]
     * @param Fz Normal load [N]
     * @return Combined force vector (Fx, Fy)
     */
    Force2D computeCombinedForce(double slip_angle, double slip_ratio, double Fz) const;

    // ========================================================================
    // Inverse Tire Model (Force → Slip)
    // ========================================================================

    /**
     * @brief Compute required slip angle for desired lateral force
     * 
     * Inverts the lateral force model to find the slip angle needed
     * to generate a target force. Uses Newton-Raphson iteration.
     * 
     * @param Fy_des Desired lateral force [N]
     * @param Fz Normal load [N]
     * @return Required slip angle [rad], clamped to feasible range
     */
    double computeRequiredSlipAngle(double Fy_des, double Fz) const;

    /**
     * @brief Compute required slip ratio for desired longitudinal force
     * @param Fx_des Desired longitudinal force [N]
     * @param Fz Normal load [N]
     * @return Required slip ratio [-], clamped to feasible range
     */
    double computeRequiredSlipRatio(double Fx_des, double Fz) const;

    // ========================================================================
    // Friction Circle and Limits
    // ========================================================================

    /**
     * @brief Get maximum force magnitude (friction circle radius)
     * @param Fz Normal load [N]
     * @return Maximum force magnitude [N]
     */
    double getMaxForce(double Fz) const {
        return mu_current_ * Fz;
    }

    /**
     * @brief Check if force is within friction circle
     * @param Fx Longitudinal force [N]
     * @param Fy Lateral force [N]
     * @param Fz Normal load [N]
     * @return true if force is feasible
     */
    bool isWithinFrictionCircle(double Fx, double Fy, double Fz) const {
        double F_max = getMaxForce(Fz);
        return (Fx * Fx + Fy * Fy) <= (F_max * F_max * 1.01); // 1% tolerance
    }

    /**
     * @brief Scale force to fit within friction circle
     * @param force Force vector to scale
     * @param Fz Normal load [N]
     * @return Scaled force within friction limits
     */
    Force2D saturateForce(const Force2D& force, double Fz) const;

    /**
     * @brief Compute friction utilization ratio
     * @param Fx Longitudinal force [N]
     * @param Fy Lateral force [N]
     * @param Fz Normal load [N]
     * @return Utilization ratio [0, 1+] (>1 means beyond friction limit)
     */
    double getFrictionUtilization(double Fx, double Fy, double Fz) const {
        double F_max = getMaxForce(Fz);
        return (F_max > EPSILON) ? std::hypot(Fx, Fy) / F_max : 0.0;
    }

    // ========================================================================
    // Linearized Model (for analysis/control design)
    // ========================================================================

    /**
     * @brief Get linearized cornering stiffness at current operating point
     * 
     * Computes ∂Fy/∂α at given slip angle. Useful for stability analysis
     * and adaptive control.
     * 
     * @param slip_angle Current slip angle [rad]
     * @param Fz Normal load [N]
     * @return Local cornering stiffness [N/rad]
     */
    double getLocalCorneringStiffness(double slip_angle, double Fz) const;

    /**
     * @brief Get linearized longitudinal stiffness
     * @param slip_ratio Current slip ratio [-]
     * @param Fz Normal load [N]
     * @return Local longitudinal stiffness [N/-]
     */
    double getLocalLongStiffness(double slip_ratio, double Fz) const;

    // ========================================================================
    // Slip Computation Utilities
    // ========================================================================

    /**
     * @brief Compute slip angle from wheel velocity
     * 
     * α = atan2(vy_wheel, |vx_wheel|)
     * 
     * @param vx_wheel Wheel velocity in x direction [m/s]
     * @param vy_wheel Wheel velocity in y direction [m/s]
     * @return Slip angle [rad]
     */
    static double computeSlipAngle(double vx_wheel, double vy_wheel);

    /**
     * @brief Compute slip ratio from wheel speed and velocity
     * 
     * For driving: κ = (ω*r - vx) / max(vx, ε)
     * For braking: κ = (ω*r - vx) / max(ω*r, ε)
     * 
     * @param wheel_speed Wheel angular velocity [rad/s]
     * @param vx_wheel Wheel translational velocity [m/s]
     * @param tire_radius Tire effective radius [m]
     * @return Slip ratio [-]
     */
    static double computeSlipRatio(double wheel_speed, double vx_wheel, double tire_radius);

private:
    TireParams params_;
    
    // Current estimated values (can be updated online)
    double mu_current_;       // Current friction coefficient
    double C_alpha_current_;  // Current cornering stiffness

    /**
     * @brief Pacejka Magic Formula core computation
     * @param x Input (slip angle or ratio)
     * @param B Stiffness factor
     * @param C Shape factor
     * @param D Peak factor
     * @param E Curvature factor
     * @return Normalized force output
     */
    static double magicFormula(double x, double B, double C, double D, double E);

    /**
     * @brief Derivative of Magic Formula (for Newton-Raphson)
     */
    static double magicFormulaDerivative(double x, double B, double C, double D, double E);
};

// ============================================================================
// Wheel-Level Kinematics
// ============================================================================

/**
 * @brief Compute wheel hub velocity from body velocity
 * 
 * Given the body velocity (vx, vy, omega) and wheel position,
 * computes the velocity at the wheel hub in the body frame.
 * 
 * v_wheel = v_body + omega × r_wheel
 * 
 * @param body_vel Body-frame velocity
 * @param wheel_pos Wheel position relative to CoG
 * @return Wheel hub velocity in body frame
 */
inline BodyVelocity computeWheelVelocity(
    const BodyVelocity& body_vel,
    const Position2D& wheel_pos)
{
    BodyVelocity wheel_vel;
    wheel_vel.vx = body_vel.vx - body_vel.omega * wheel_pos.y;
    wheel_vel.vy = body_vel.vy + body_vel.omega * wheel_pos.x;
    wheel_vel.omega = body_vel.omega;
    return wheel_vel;
}

/**
 * @brief Compute wheel velocity in tire frame (after steering)
 * 
 * Rotates the wheel hub velocity by the steering angle to get
 * the velocity components in the tire's local frame.
 * 
 * @param wheel_vel Wheel hub velocity in body frame
 * @param steer_angle Steering angle [rad]
 * @return Velocity in tire frame (vx_tire, vy_tire)
 */
inline std::pair<double, double> transformToTireFrame(
    const BodyVelocity& wheel_vel,
    double steer_angle)
{
    double cos_d = std::cos(steer_angle);
    double sin_d = std::sin(steer_angle);
    double vx_tire = wheel_vel.vx * cos_d + wheel_vel.vy * sin_d;
    double vy_tire = -wheel_vel.vx * sin_d + wheel_vel.vy * cos_d;
    return {vx_tire, vy_tire};
}

/**
 * @brief Transform tire force to body frame
 * 
 * @param tire_force Force in tire frame
 * @param steer_angle Steering angle [rad]
 * @return Force in body frame
 */
inline Force2D transformForceToBodyFrame(
    const Force2D& tire_force,
    double steer_angle)
{
    double cos_d = std::cos(steer_angle);
    double sin_d = std::sin(steer_angle);
    Force2D body_force;
    body_force.Fx = tire_force.Fx * cos_d - tire_force.Fy * sin_d;
    body_force.Fy = tire_force.Fx * sin_d + tire_force.Fy * cos_d;
    return body_force;
}

} // namespace mppi_tf
