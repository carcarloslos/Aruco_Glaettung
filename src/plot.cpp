#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <eigen3/Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> linspaced;
    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // Ensure that start and end are exactly the same as the input

    return linspaced;
}

int main() {
    // Erzeugen wilderer Originaldaten
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.2);
    
    std::vector<double> timestamps = linspace(0, 1, 30);
    std::vector<double> positions;

    for (double t : timestamps) {
        positions.push_back(std::sin(10 * 2 * M_PI * t) + 0.5 * std::cos(25 * 2 * M_PI * t) + d(gen));
    }

    // Interpolation auf 100 Hz
    std::vector<double> target_timestamps = linspace(0, 1, 100);
    Eigen::VectorXd interp_positions = Eigen::VectorXd::Zero(100);

    for (int i = 0; i < target_timestamps.size(); ++i) {
        double t = target_timestamps[i];
        int idx = std::lower_bound(timestamps.begin(), timestamps.end(), t) - timestamps.begin();
        if (idx == 0) {
            interp_positions[i] = positions[0];
        } else if (idx >= timestamps.size()) {
            interp_positions[i] = positions.back();
        } else {
            double t1 = timestamps[idx - 1];
            double t2 = timestamps[idx];
            double p1 = positions[idx - 1];
            double p2 = positions[idx];
            interp_positions[i] = p1 + (p2 - p1) * (t - t1) / (t2 - t1);
        }
    }

    // Kalman-Filter-Setup
    int n_iter = target_timestamps.size();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n_iter);
    Eigen::VectorXd P = Eigen::VectorXd::Ones(n_iter);
    double Q = 1e-5;  // Prozessrauschen-Kovarianz
    double R = 1e-2;  // Messrauschen-Kovarianz
    Eigen::VectorXd x_hat = Eigen::VectorXd::Zero(n_iter);
    Eigen::VectorXd P_hat = Eigen::VectorXd::Zero(n_iter);
    Eigen::VectorXd K = Eigen::VectorXd::Zero(n_iter);

    // Initialisierung
    x[0] = interp_positions[0];
    P[0] = 1.0;

    // Kalman-Filter-Durchlauf
    for (int k = 1; k < n_iter; ++k) {
        // Vorhersage
        x_hat[k] = x[k - 1];
        P_hat[k] = P[k - 1] + Q;

        // Update
        K[k] = P_hat[k] / (P_hat[k] + R);
        x[k] = x_hat[k] + K[k] * (interp_positions[k] - x_hat[k]);
        P[k] = (1 - K[k]) * P_hat[k];
    }

    // Ergebnisse anzeigen
    plt::plot(timestamps, positions, "bo", {{"label", "Originaldaten (30 Hz)"}});
    plt::plot(target_timestamps, std::vector<double>(interp_positions.data(), interp_positions.data() + interp_positions.size()), "r.", {{"label", "Interpolierte Daten (100 Hz)"}});
    plt::plot(target_timestamps, std::vector<double>(x.data(), x.data() + x.size()), "g-", {{"label", "Gefilterte Daten (Kalman)"}});
    plt::legend();
    plt::xlabel("Zeit (s)");
    plt::ylabel("Position");
    plt::title("Interpolation und Kalman-Filter zur Positionsschätzung");
    plt::show();

    return 0;
}

