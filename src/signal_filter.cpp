#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
//KEINE AHNUNG LOL
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> linspaced;
    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); 

    return linspaced;
}

int main() {
    double freq_main = 1.0; // Main frequency for smooth signal
    double noise_amplitude = 1.0; // Amplitude of noise spikes
    double noise_probability = 0.1; // Probability of a noise spike

    std::vector<double> timestamps = linspace(0, 1, 100);
    std::vector<double> true_positions;
    std::vector<double> noisy_positions;

    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise_dist(-noise_amplitude, noise_amplitude);
    std::uniform_real_distribution<> spike_prob(0.0, 1.0);

    for (double t : timestamps) {
        // Smooth sinusoidal function
        double true_value = std::sin(2 * M_PI * freq_main * t);
        true_positions.push_back(true_value);

        // Add occasional noise spikes
        double noisy_value = true_value;
        if (spike_prob(gen) < noise_probability) {
            noisy_value += noise_dist(gen); // Add a random noise spike
        }
        noisy_positions.push_back(noisy_value);
    }

    // Kalman Filter Setup
    int n_iter = timestamps.size();
    Eigen::VectorXd x(n_iter);
    Eigen::VectorXd P(n_iter);
    Eigen::VectorXd Q(n_iter); // Process noise covariance
    Eigen::VectorXd R(n_iter); // Measurement noise covariance
    Eigen::VectorXd x_hat(n_iter); // a priori estimate
    Eigen::VectorXd P_hat(n_iter); // a priori covariance
    Eigen::VectorXd K(n_iter); // Kalman gain

    double Q_value = 1e-5; // Process noise covariance value
    double R_value = 1.0; // Measurement noise covariance value

    x.setZero();
    P.setOnes();
    Q.setConstant(Q_value);
    R.setConstant(R_value);

    // Initial estimate
    x[0] = noisy_positions[0];
    P[0] = 1.0;

    // Kalman Filter
    for (int k = 1; k < n_iter; ++k) {
        // Prediction step
        x_hat[k] = x[k - 1];
        P_hat[k] = P[k - 1] + Q[k];

        // Update step
        K[k] = P_hat[k] / (P_hat[k] + R[k]);
        x[k] = x_hat[k] + K[k] * (noisy_positions[k] - x_hat[k]);
        P[k] = (1 - K[k]) * P_hat[k];
    }

    // Plot results


    return 0;
}
