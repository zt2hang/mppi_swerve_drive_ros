/**
 * @file tire_model.cpp
 * @brief Implementation of Pacejka-based tire force model
 * 
 * @author ZZT
 * @date 2024
 */

#include "mppi_tf/tire_model.hpp"
#include <cmath>
#include <algorithm>

namespace mppi_tf
{

TireModel::TireModel(const TireParams& params)
    : params_(params)
    , mu_current_(params.mu_peak)
    , C_alpha_current_(params.C_alpha)
{
}

// ============================================================================
// Pacejka Magic Formula Core
// ============================================================================

double TireModel::magicFormula(double x, double B, double C, double D, double E)
{
    // F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
    double Bx = B * x;
    double inner = Bx - E * (Bx - std::atan(Bx));
    return D * std::sin(C * std::atan(inner));
}

double TireModel::magicFormulaDerivative(double x, double B, double C, double D, double E)
{
    // Derivative of Magic Formula with respect to x
    // d/dx[D * sin(C * atan(Bx - E*(Bx - atan(Bx))))]
    
    double Bx = B * x;
    double atan_Bx = std::atan(Bx);
    double inner = Bx - E * (Bx - atan_Bx);
    double atan_inner = std::atan(inner);
    
    // d(inner)/dx
    double d_atan_Bx = B / (1.0 + Bx * Bx);
    double d_inner = B - E * (B - d_atan_Bx);
    
    // d(atan(inner))/dx
    double d_atan_inner = d_inner / (1.0 + inner * inner);
    
    // Final derivative
    return D * C * std::cos(C * atan_inner) * d_atan_inner;
}

// ============================================================================
// Pure Slip Force Computation
// ============================================================================

double TireModel::computeLateralForce(double slip_angle, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    // Peak force
    double D = mu_current_ * Fz;
    
    // Stiffness factor: B = C_alpha / (C * D) at origin
    // This ensures the initial slope matches C_alpha
    double C = params_.C;
    double B = C_alpha_current_ / (C * D + EPSILON);
    
    // Curvature factor
    double E = params_.E;
    
    // Apply Magic Formula
    // Note: slip_angle positive means force in negative y (towards turn center)
    double Fy = magicFormula(slip_angle, B, C, D, E);
    
    // The Magic Formula gives force opposing the slip, so we negate
    return -Fy;
}

double TireModel::computeLongitudinalForce(double slip_ratio, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    // Peak force (same friction coefficient)
    double D = mu_current_ * Fz;
    
    // For longitudinal, typically higher stiffness
    double C_kappa = params_.C_kappa;
    double C = 1.65;  // Shape factor for longitudinal (typically different from lateral)
    double B = C_kappa / (C * D + EPSILON);
    
    // Curvature factor (often closer to 1 for longitudinal)
    double E = 0.95;
    
    // Apply Magic Formula
    return magicFormula(slip_ratio, B, C, D, E);
}

// ============================================================================
// Combined Slip Force Computation
// ============================================================================

Force2D TireModel::computeCombinedForce(double slip_angle, double slip_ratio, double Fz) const
{
    Force2D force;
    
    if (Fz < EPSILON) {
        return force;
    }
    
    // Check for pure slip cases
    double alpha_abs = std::abs(slip_angle);
    double kappa_abs = std::abs(slip_ratio);
    
    if (alpha_abs < 1e-4 && kappa_abs < 1e-4) {
        // No slip - no force
        return force;
    }
    
    // Compute pure slip forces first
    double Fy0 = computeLateralForce(slip_angle, Fz);
    double Fx0 = computeLongitudinalForce(slip_ratio, Fz);
    
    // If only one type of slip, return pure slip force
    if (kappa_abs < 1e-4) {
        force.Fy = Fy0;
        return force;
    }
    if (alpha_abs < 1e-4) {
        force.Fx = Fx0;
        return force;
    }
    
    // Combined slip: use friction ellipse method
    // Normalized slip quantities
    double kappa_norm = slip_ratio / (params_.kappa_peak + EPSILON);
    double alpha_norm = std::tan(slip_angle) / (std::tan(params_.alpha_peak) + EPSILON);
    
    // Combined slip magnitude
    double sigma = std::sqrt(kappa_norm * kappa_norm + alpha_norm * alpha_norm);
    
    if (sigma < EPSILON) {
        return force;
    }
    
    // Scale forces by their contribution to combined slip
    // This naturally enforces the friction ellipse constraint
    force.Fx = Fx0 * std::abs(kappa_norm) / sigma;
    force.Fy = Fy0 * std::abs(alpha_norm) / sigma;
    
    // Ensure we don't exceed friction circle
    double F_max = mu_current_ * Fz;
    double F_total = std::hypot(force.Fx, force.Fy);
    
    if (F_total > F_max) {
        double scale = F_max / F_total;
        force.Fx *= scale;
        force.Fy *= scale;
    }
    
    return force;
}

// ============================================================================
// Inverse Tire Model
// ============================================================================

double TireModel::computeRequiredSlipAngle(double Fy_des, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    // Maximum achievable force
    double Fy_max = mu_current_ * Fz;
    
    // Clamp desired force to achievable range
    double Fy_target = std::clamp(Fy_des, -Fy_max * 0.99, Fy_max * 0.99);
    
    // For small forces, use linear approximation
    if (std::abs(Fy_target) < 0.1 * Fy_max) {
        return -Fy_target / (C_alpha_current_ + EPSILON);
    }
    
    // Newton-Raphson iteration for inverse
    double alpha = -Fy_target / (C_alpha_current_ + EPSILON); // Initial guess
    
    // Pacejka parameters
    double D = mu_current_ * Fz;
    double C = params_.C;
    double B = C_alpha_current_ / (C * D + EPSILON);
    double E = params_.E;
    
    for (int i = 0; i < 10; ++i) {
        double Fy_current = -magicFormula(alpha, B, C, D, E);
        double error = Fy_current - Fy_target;
        
        if (std::abs(error) < 1e-3) break;
        
        double dFy_dalpha = -magicFormulaDerivative(alpha, B, C, D, E);
        
        if (std::abs(dFy_dalpha) < EPSILON) break;
        
        alpha -= error / dFy_dalpha;
        
        // Clamp to reasonable range
        alpha = std::clamp(alpha, -0.3, 0.3);
    }
    
    return alpha;
}

double TireModel::computeRequiredSlipRatio(double Fx_des, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    // Maximum achievable force
    double Fx_max = mu_current_ * Fz;
    
    // Clamp desired force
    double Fx_target = std::clamp(Fx_des, -Fx_max * 0.99, Fx_max * 0.99);
    
    // For small forces, use linear approximation
    if (std::abs(Fx_target) < 0.1 * Fx_max) {
        return Fx_target / (params_.C_kappa + EPSILON);
    }
    
    // Newton-Raphson iteration
    double kappa = Fx_target / (params_.C_kappa + EPSILON);
    
    double D = mu_current_ * Fz;
    double C_long = 1.65;
    double B = params_.C_kappa / (C_long * D + EPSILON);
    double E = 0.95;
    
    for (int i = 0; i < 10; ++i) {
        double Fx_current = magicFormula(kappa, B, C_long, D, E);
        double error = Fx_current - Fx_target;
        
        if (std::abs(error) < 1e-3) break;
        
        double dFx_dkappa = magicFormulaDerivative(kappa, B, C_long, D, E);
        
        if (std::abs(dFx_dkappa) < EPSILON) break;
        
        kappa -= error / dFx_dkappa;
        kappa = std::clamp(kappa, -0.5, 0.5);
    }
    
    return kappa;
}

// ============================================================================
// Friction Circle Operations
// ============================================================================

Force2D TireModel::saturateForce(const Force2D& force, double Fz) const
{
    double F_max = getMaxForce(Fz);
    double F_mag = force.magnitude();
    
    if (F_mag <= F_max || F_mag < EPSILON) {
        return force;
    }
    
    // Scale to friction circle
    double scale = F_max / F_mag;
    return Force2D{force.Fx * scale, force.Fy * scale};
}

// ============================================================================
// Linearized Model
// ============================================================================

double TireModel::getLocalCorneringStiffness(double slip_angle, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    double D = mu_current_ * Fz;
    double C = params_.C;
    double B = C_alpha_current_ / (C * D + EPSILON);
    double E = params_.E;
    
    // Return magnitude of derivative (cornering stiffness is positive)
    return std::abs(magicFormulaDerivative(slip_angle, B, C, D, E));
}

double TireModel::getLocalLongStiffness(double slip_ratio, double Fz) const
{
    if (Fz < EPSILON) return 0.0;
    
    double D = mu_current_ * Fz;
    double C_long = 1.65;
    double B = params_.C_kappa / (C_long * D + EPSILON);
    double E = 0.95;
    
    return std::abs(magicFormulaDerivative(slip_ratio, B, C_long, D, E));
}

// ============================================================================
// Slip Computation Utilities
// ============================================================================

double TireModel::computeSlipAngle(double vx_wheel, double vy_wheel)
{
    // Slip angle is the angle between wheel heading and velocity direction
    // α = atan2(vy, vx) when vx > 0
    // Handle low speed case
    double v_mag = std::hypot(vx_wheel, vy_wheel);
    
    if (v_mag < 0.1) {
        // At very low speeds, slip angle is undefined/unstable
        return 0.0;
    }
    
    // Use atan2 for proper quadrant handling
    // Positive slip angle means velocity is to the right of wheel heading
    return std::atan2(vy_wheel, std::abs(vx_wheel) + EPSILON);
}

double TireModel::computeSlipRatio(double wheel_speed, double vx_wheel, double tire_radius)
{
    // Slip ratio: κ = (v_wheel - v_ground) / max(|v_wheel|, |v_ground|)
    double v_tire = wheel_speed * tire_radius;  // Linear velocity from wheel rotation
    double v_ground = vx_wheel;                  // Ground speed
    
    // Handle low speed case
    if (std::abs(v_tire) < 0.1 && std::abs(v_ground) < 0.1) {
        return 0.0;
    }
    
    // Normalization factor (prevents division by zero and singularity)
    double v_norm = std::max(std::abs(v_tire), std::abs(v_ground));
    v_norm = std::max(v_norm, 0.1);  // Minimum normalization
    
    // Slip ratio
    // Positive = driving (wheel spinning faster than ground speed)
    // Negative = braking (wheel slower than ground speed)
    double kappa = (v_tire - v_ground) / v_norm;
    
    // Clamp to physically reasonable range
    return std::clamp(kappa, -1.0, 1.0);
}

} // namespace mppi_tf
