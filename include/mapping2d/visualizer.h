#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sophus/se2.hpp>

#include "Submap.h"

namespace Visualizer {

void Visualize2dScan(const sensor_msgs::msg::LaserScan scan,
                     const Sophus::SE2d &Tlf, cv::Mat &img,
                     cv::Vec3b scan_color, bool show_robot = true,
                     const int img_size = 800, const double resolution = 20.0,
                     const Sophus::SE2d &Twl = Sophus::SE2d());

cv::Mat global_map(const std::vector<Submap *> &submaps,
                   const int map_dim = 1000);

}; // namespace Visualizer