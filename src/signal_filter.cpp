#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Dense>
#include <unordered_set>


struct KalmanFilterState {
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::VectorXd x_hat;
    Eigen::MatrixXd P_hat;
    Eigen::VectorXd K;
    int n_iter;

    // Default constructor
    KalmanFilterState() : n_iter(0) {
        x = Eigen::VectorXd(7);
        P = Eigen::MatrixXd::Identity(7, 7);
        Q = Eigen::MatrixXd::Identity(7, 7);
        R = Eigen::MatrixXd::Identity(7, 7);
        x_hat = Eigen::VectorXd(7);
        P_hat = Eigen::MatrixXd(7, 7);
        K = Eigen::VectorXd(7);
    }

    // Parameterized constructor
    KalmanFilterState(int n_iter, double Q_value, double R_value) : n_iter(n_iter) {
        x = Eigen::VectorXd(7);
        P = Eigen::MatrixXd::Identity(7, 7);
        Q = Eigen::MatrixXd::Identity(7, 7) * Q_value;
        R = Eigen::MatrixXd::Identity(7, 7) * R_value;
        x_hat = Eigen::VectorXd(7);
        P_hat = Eigen::MatrixXd(7, 7);
        K = Eigen::VectorXd(7);
        x.setZero();
        P = Eigen::MatrixXd::Identity(7, 7);
    }
};


// Global variables
std::unordered_set<int> known_ids = {54, 55, 53, 57};  
int max_iterations = 150; 
std::unordered_map<int, KalmanFilterState> kalman_filters;
std::unordered_map<int, visualization_msgs::Marker> filtered_markers;
bool data_received = false;

// Normalize a quaternion
void normalizeQuaternion(Eigen::Vector4d& q) {
    q.normalize();
}

// Callback function for the MarkerArray subscriber
void markerArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
    for (const auto& marker : msg->markers) {
        int marker_id = marker.id;
        if (known_ids.find(marker_id) != known_ids.end()){
        if (kalman_filters.find(marker_id) == kalman_filters.end()) {
        
        
            // Initialize Kalman filter for this marker if not already present
            
            
            kalman_filters.emplace(marker_id, KalmanFilterState(max_iterations, 3e-1, 3.0));
        }

        KalmanFilterState& kf = kalman_filters[marker_id];

        // Prediction step
        kf.x_hat.head(3) = kf.x.head(3); // Position
        kf.x_hat.tail(4) = kf.x.tail(4); // Orientation
        kf.P_hat = kf.P + kf.Q;

        // Update step for position
        Eigen::Vector3d measurement_pos(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z);
        Eigen::Vector3d predicted_pos = kf.x_hat.head(3);
        Eigen::MatrixXd P_hat_pos = kf.P_hat.block<3, 3>(0, 0);
        Eigen::MatrixXd R_pos = kf.R.block<3, 3>(0, 0);
        Eigen::MatrixXd K_pos = P_hat_pos * (P_hat_pos + R_pos).inverse();

        kf.x.head(3) = predicted_pos + K_pos * (measurement_pos - predicted_pos);
        kf.P.block<3, 3>(0, 0) = (Eigen::MatrixXd::Identity(3, 3) - K_pos) * P_hat_pos;

        // Update step for orientation
        Eigen::Vector4d measurement_ori(marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w);
        normalizeQuaternion(measurement_ori);  // Normalize the incoming quaternion
        Eigen::Vector4d predicted_ori = kf.x_hat.tail(4);
        Eigen::MatrixXd P_hat_ori = kf.P_hat.block<4, 4>(3, 3);
        Eigen::MatrixXd R_ori = kf.R.block<4, 4>(3, 3);
        Eigen::MatrixXd K_ori = P_hat_ori * (P_hat_ori + R_ori).inverse();

        Eigen::Vector4d updated_ori = predicted_ori + K_ori * (measurement_ori - predicted_ori);
        normalizeQuaternion(updated_ori);  // Ensure the state quaternion remains normalized
        kf.x.tail(4) = updated_ori;
        kf.P.block<4, 4>(3, 3) = (Eigen::MatrixXd::Identity(4, 4) - K_ori) * P_hat_ori;

        // Create a new marker with the filtered position and orientation
        filtered_markers[marker_id] = marker;
        filtered_markers[marker_id].pose.position.x = kf.x[0];
        filtered_markers[marker_id].pose.position.y = kf.x[1];
        filtered_markers[marker_id].pose.position.z = kf.x[2];
        filtered_markers[marker_id].pose.orientation.x = kf.x[3];
        filtered_markers[marker_id].pose.orientation.y = kf.x[4];
        filtered_markers[marker_id].pose.orientation.z = kf.x[5];
        filtered_markers[marker_id].pose.orientation.w = kf.x[6];
    }
    data_received = true;
}}


int main(int argc, char** argv) {

    ros::init(argc, argv, "kalman_filter_node");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/visualization_marker_array", 1000, markerArrayCallback);

    ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/filtered_marker_array", 10);

    ros::Rate rate(100); 
    while (ros::ok()) {
    	
        if (data_received) {
            visualization_msgs::MarkerArray marker_array_msg;
            for (const auto& kv : filtered_markers) {
                marker_array_msg.markers.push_back(kv.second);
            }
            marker_pub.publish(marker_array_msg);
        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}


