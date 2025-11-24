#pragma once

#include <Eigen/Dense>
#include <deque>
#include <algorithm>
#include <iostream>

namespace mppi_h_adaptive
{

class AdaptiveEstimator
{
public:
    // Constructor
    AdaptiveEstimator(int input_dim = 11, int output_dim = 3, int hidden_dim = 32, double forgetting_factor = 0.99)
        : lambda_(forgetting_factor) // Use learning_rate as forgetting factor (0.95 - 0.999)
    {
        // Initialize Estimator
        // Model: v_actual = K * v_cmd
        // K is 3x3 matrix. Initially Identity (no slip).
        K_ = Eigen::MatrixXd::Identity(3, 3);
        
        // Covariance Matrix P
        // Large initial covariance means "uncertain", allows fast initial adaptation
        P_ = Eigen::MatrixXd::Identity(3, 3) * 1000.0;
    }

    // Forward pass: Predicts the RESIDUAL (v_actual - v_cmd)
    // So that: v_corrected = v_cmd + residual
    // Since our model is v_actual = K * v_cmd
    // residual = K * v_cmd - v_cmd = (K - I) * v_cmd
    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        // Input is [vx, vy, w, ...wheel_params...]
        // We only use the first 3 (body velocity command)
        Eigen::Vector3d v_cmd;
        v_cmd << input(0), input(1), input(2);
        
        Eigen::Vector3d v_est = K_ * v_cmd;
        
        // Return residual
        return v_est - v_cmd;
    }

    // Train: Update the estimator K using RLS
    // input: v_cmd (and others)
    // target: v_actual - v_cmd (the residual error)
    double train(const Eigen::VectorXd& input, const Eigen::VectorXd& target)
    {
        // Reconstruct v_actual and v_cmd
        Eigen::Vector3d v_cmd;
        v_cmd << input(0), input(1), input(2);
        
        Eigen::Vector3d v_residual_measured;
        v_residual_measured << target(0), target(1), target(2);
        
        Eigen::Vector3d v_actual = v_cmd + v_residual_measured;
        
        // RLS Update for each row of K independently (or as a block)
        // y = K * x
        // We want to find K.
        // Transpose: y^T = x^T * K^T
        // Let theta_i be the i-th row of K (as a column vector).
        // y_i = x^T * theta_i
        
        // Common x vector for all outputs
        Eigen::Vector3d x = v_cmd;
        
        // Avoid division by zero or updating on zero input
        if (x.squaredNorm() < 1e-4) return 0.0;

        // Calculate Gain vector k (3x1)
        // k = P * x / (lambda + x^T * P * x)
        Eigen::Vector3d k = P_ * x / (lambda_ + x.transpose() * P_ * x);
        
        // Update P
        // P = (P - k * x^T * P) / lambda
        P_ = (P_ - k * x.transpose() * P_) / lambda_;
        
        // Update K (each row)
        // Error vector e = y - K * x
        Eigen::Vector3d y_est = K_ * x;
        Eigen::Vector3d e = v_actual - y_est;
        
        // K_new = K_old + e * k^T
        K_ += e * k.transpose();
        
        // Stability clamping:
        // Diagonal elements (gains) should be within reasonable bounds [0.1, 1.5]
        // Off-diagonal elements should be small [-0.5, 0.5]
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                if (i==j) {
                    K_(i,j) = std::max(0.1, std::min(1.5, K_(i,j)));
                } else {
                    K_(i,j) = std::max(-0.5, std::min(0.5, K_(i,j)));
                }
            }
        }
        
        return e.norm(); // Return prediction error magnitude
    }

    // Helper to get input vector from MPPI variables
    static Eigen::VectorXd prepareInput(double vx, double vy, double w, const std::vector<double>& wheel_params)
    {
        Eigen::VectorXd input(11);
        input(0) = vx;
        input(1) = vy;
        input(2) = w;
        // Wheel params are ignored in this simple adaptive model, but kept for interface compatibility
        for(int i=0; i<8; ++i) {
            input(3+i) = wheel_params[i];
        }
        return input;
    }

private:
    double lambda_; // Forgetting factor
    Eigen::MatrixXd K_; // Estimated Slip Matrix (3x3)
    Eigen::MatrixXd P_; // Covariance Matrix (3x3)
};

} // namespace mppi_h_adaptive
