#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mppi_h
{

class TireStiffnessEstimator
{
public:
    TireStiffnessEstimator()
    {
        // Initial guess: 0.0 (No slip / Infinite stiffness)
        // K_slip represents the compliance (inverse of stiffness).
        // Unit: (m/s) / (m/s^2) = s ? Or dimensionless if normalized?
        // Model: v_slip_y = - K_slip * (vx * omega)
        slip_factor_ = 0.0; 
    }

    // Update the estimator based on measured data
    // vx_cmd, omega_cmd: Command velocities
    // vy_act: Actual lateral velocity (from odom)
    // vy_cmd: Command lateral velocity (usually 0 for differential drive, but non-zero for swerve)
    void update(const double vx_cmd, const double vy_cmd, const double omega_cmd,
                const double vx_act, const double vy_act, const double omega_act,
                double dt)
    {
        // 1. Calculate Lateral Acceleration Demand (Centripetal term)
        // a_lat_demand = v_x * omega
        // This is the main driver for lateral slip during cornering.
        double a_lat = vx_cmd * omega_cmd;

        // Avoid updating on straight lines or static (low observability of stiffness)
        // Threshold: 0.1 m/s^2
        if (std::abs(a_lat) < 0.1) return;

        // 2. Calculate Lateral Velocity Error (Observed Slip)
        // We assume the error in Vy is dominated by tire slip.
        double vy_error = vy_act - vy_cmd;

        // 3. Model Prediction: 
        // vy_slip_pred = - slip_factor * a_lat
        double vy_slip_pred = - slip_factor_ * a_lat;
        
        // 4. Residual (Prediction Error)
        // residual = Actual_Error - Predicted_Slip
        double residual = vy_error - vy_slip_pred;
        
        // 5. Gradient Descent Update
        // We want to minimize J = 0.5 * residual^2
        // dJ/dK = residual * d(residual)/dK
        // residual = vy_error - (-K * a_lat) = vy_error + K * a_lat
        // d(residual)/dK = a_lat
        // K_new = K_old - learning_rate * residual * a_lat
        
        double learning_rate = 0.01; // Reduced from 0.05 to improve stability
        
        // Normalize learning rate by a_lat magnitude to avoid instability at high speeds
        // effective_lr = lr / (a_lat^2 + epsilon) ? 
        // Let's stick to simple SGD with small rate for now.
        
        slip_factor_ -= learning_rate * residual * a_lat;
        
        // 6. Constraints
        // Slip factor must be positive (slip opposes force)
        // And bounded (e.g., ice surface limit)
        // Reduced max limit to 0.3 to prevent over-estimation on tiles
        slip_factor_ = std::max(0.0, std::min(0.3, slip_factor_));
        
        // Debug output occasionally
        static int count = 0;
        if (count++ % 20 == 0) {
            std::cout << "[StiffnessEstimator] K_slip: " << slip_factor_ 
                      << " | Res: " << residual 
                      << " | a_lat: " << a_lat 
                      << " | vy_err: " << vy_error << std::endl;
        }
    }

    // Get the estimated slip factor for use in MPPI prediction
    double getSlipFactor() const
    {
        return slip_factor_;
    }

    // Reset estimator (e.g. when robot stops)
    void reset()
    {
        slip_factor_ = 0.0;
    }

private:
    double slip_factor_; // Estimated slip coefficient
};

} // namespace mppi_h
