#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <functional>

#include "Frame.h"

class OccupancyGridMap {
public:
  OccupancyGridMap(const int img_size, const double resolution)
      : img_size_(img_size), img_offset_(0.5 * img_size_),
        resolution_(resolution),
        map_(cv::Mat(img_size, img_size, CV_8UC1, 127)) {}

  bool add_frame(const Frame &frame);
  bool add_frame(const sensor_msgs::msg::LaserScan &scan,
                 const Sophus::SE2d &pose);
  void draw_line(const Eigen::Vector2i start, const Eigen::Vector2i end);
  void bresenham(const Eigen::Vector2i start, const Eigen::Vector2i end,
                 std::function<void(const int u, const int v)> free_cell);
  bool set_cell(const bool free_cell, const int x, const int y);
  void reset();

  cv::Mat binary_map(const int occupied_threshold = 117,
                     const int free_threshold = 137) const;
  cv::Mat map() const { return map_; };
  void set_map(cv::Mat map) { map_ = map.clone(); };

  bool inside(const int r, const int c) const;
  bool has_outsider() const { return outsider_; };

private:
  int img_size_ = 0;
  int img_offset_ = 0;
  double resolution_ = 0.0;
  cv::Mat map_;
  bool outsider_ = false;
};