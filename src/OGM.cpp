#include "OGM.h"

#include <glog/logging.h>
#include <sensor_msgs/msg/laser_scan.hpp>

void OccupancyGridMap::reset() {
  map_.release();
  map_ = cv::Mat(img_size_, img_size_, CV_8UC1, 127);
  outsider_ = false;
}

bool OccupancyGridMap::inside(const int r, const int c) const {
  return r >= 0 && r < map_.rows && c >= 0 && c < map_.cols;
}

bool OccupancyGridMap::add_frame(const sensor_msgs::msg::LaserScan &scan,
                                 const Sophus::SE2d &pose) {
  if (scan.ranges.empty()) {
    return false;
  }
  const Eigen::Vector2i coord_start(
      resolution_ * pose.translation().x() + img_offset_,
      resolution_ * pose.translation().y() + img_offset_);
  if (!inside(coord_start.y(), coord_start.x())) {
    return false;
  }

  size_t valid_cnt = 0;
  for (size_t sid = 0; sid < scan.ranges.size(); ++sid) {
    const double range = scan.ranges[sid];
    const double angle = scan.angle_min + sid * scan.angle_increment;
    if (range < scan.range_min || range > scan.range_max ||
        angle < scan.angle_min + 30 * M_PI / 180 ||
        angle > scan.angle_max - 30 * M_PI / 180) {
      continue;
    }
    const Eigen::Vector2d pos = pose * Eigen::Vector2d(range * std::cos(angle),
                                                       range * std::sin(angle));
    const Eigen::Vector2i coord_end(resolution_ * pos.x() + img_offset_,
                                    resolution_ * pos.y() + img_offset_);
    if (!inside(coord_end.y(), coord_end.x())) {
      outsider_ = true;
      continue;
    }
    draw_line(coord_start, coord_end);
    valid_cnt++;
  }
  return valid_cnt > 0;
}

bool OccupancyGridMap::add_frame(const Frame &frame) {
  return add_frame(frame.scan(), frame.Tlf());
}

void OccupancyGridMap::draw_line(const Eigen::Vector2i start,
                                 const Eigen::Vector2i end) {
  if (start == end) {
    return;
  }
  const int dx = end.x() - start.x();
  const int dy = end.y() - start.y();

  if (dx >= 0 && dy >= 0) {
    if (dx > dy) {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, u, v);
      };
      bresenham(start, end, free_cell);
    } else {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, v, u);
      };
      bresenham(Eigen::Vector2i(start.y(), start.x()),
                Eigen::Vector2i(end.y(), end.x()), free_cell);
    }
  } else if (dx <= 0 && dy <= 0) {
    if (-dx > -dy) {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, -u, -v);
      };
      bresenham(-start, -end, free_cell);
    } else {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, -v, -u);
      };
      bresenham(Eigen::Vector2i(-start.y(), -start.x()),
                Eigen::Vector2i(-end.y(), -end.x()), free_cell);
    }
  } else if (dx < 0 && dy > 0) {
    if (-dx > dy) {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, -u, v);
      };
      bresenham(Eigen::Vector2i(-start.x(), start.y()),
                Eigen::Vector2i(-end.x(), end.y()), free_cell);
    } else {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, -v, u);
      };
      bresenham(Eigen::Vector2i(start.y(), -start.x()),
                Eigen::Vector2i(end.y(), -end.x()), free_cell);
    }
  } else if (dx > 0 && dy < 0) {
    if (dx > -dy) {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, u, -v);
      };
      bresenham(Eigen::Vector2i(start.x(), -start.y()),
                Eigen::Vector2i(end.x(), -end.y()), free_cell);
    } else {
      auto free_cell = [this](const int u, const int v) {
        set_cell(true, v, -u);
      };
      bresenham(Eigen::Vector2i(-start.y(), start.x()),
                Eigen::Vector2i(-end.y(), end.x()), free_cell);
    }
  }
  set_cell(false, end.x(), end.y());
}

void OccupancyGridMap::bresenham(
    const Eigen::Vector2i start, const Eigen::Vector2i end,
    std::function<void(const int u, const int v)> free_cell) {
  const double du = std::abs(start[0] - end[0]);
  const double dv = std::abs(start[1] - end[1]);
  double err = -du;

  for (int u = start[0], v = start[1]; u < end[0]; ++u) {
    free_cell(u, v);
    err += 2 * dv;
    if (err > 0) {
      err -= 2 * du;
      v++;
    }
  }
}

bool OccupancyGridMap::set_cell(const bool free_cell, const int x,
                                const int y) {
  if (x < 0 || x >= map_.cols || y < 0 || y >= map_.rows) {
    return false;
  }
  if (free_cell && map_.at<uchar>(y, x) < 137) {
    map_.at<uchar>(y, x) += 1;
  } else if (!free_cell && map_.at<uchar>(y, x) > 117) {
    map_.at<uchar>(y, x) -= 1;
  }
  return true;
}

cv::Mat OccupancyGridMap::binary_map(const int occupied_threshold,
                                     const int free_threshold) const {
  cv::Mat binary_map(map_.rows, map_.cols, CV_8UC1, 127);
  for (int r = 0; r < map_.rows; ++r) {
    for (int c = 0; c < map_.cols; ++c) {
      if (map_.at<uchar>(r, c) < occupied_threshold) {
        binary_map.at<uchar>(r, c) = 0;
      } else if (map_.at<uchar>(r, c) > free_threshold) {
        binary_map.at<uchar>(r, c) = 255;
      }
    }
  }
  return binary_map;
}
