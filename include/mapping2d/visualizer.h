#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sophus/se2.hpp>

namespace Visualizer {

void Visualize2dScan(const sensor_msgs::msg::LaserScan scan,
                     const Sophus::SE2d &Tor, cv::Mat &img,
                     cv::Vec3b scan_color, bool show_robot = true,
                     const int img_size = 800, const double resolution = 20.0,
                     const Sophus::SE2d &Tos = Sophus::SE2d());

}; // namespace Visualizer