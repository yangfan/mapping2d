#pragma once

#include "OGM.h"

#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sophus/se2.hpp>

class LikelihoodField {
public:
  enum class SolverType { GN, G2o, Ceres };
  struct LocalPatch {
    explicit LocalPatch(const int rg)
        : cell_range(rg), data(2 * rg + 1, 2 * rg + 1, CV_64FC1) {
      for (int r = 0; r < data.rows; ++r) {
        for (int c = 0; c < data.cols; ++c) {
          data.at<double>(r, c) =
              std::sqrt((r - cell_range) * (r - cell_range) +
                        (c - cell_range) * (c - cell_range));
        }
      }
    }
    double cell_value(const int delta_x, const int delta_y) const {
      return data.at<double>(delta_y + cell_range, delta_x + cell_range);
    }
    int cell_range = 0;
    cv::Mat data;
  };

  LikelihoodField(const double patch_radius, const int img_size,
                  const double resolution, const double sigma)
      : local_patch_(int(patch_radius * resolution)), img_size_(img_size),
        img_offset_(0.5 * img_size), resolution_(resolution),
        max_cell_dist_(1.5 * local_patch_.cell_range), sigma_(sigma),
        dist_map_(img_size, img_size, CV_64FC1, max_cell_dist_) {}

  void reset();

  bool set_dist_map(const sensor_msgs::msg::LaserScan &target_scan);
  bool set_dist_map(const OccupancyGridMap &grid_map);

  // scan to scan
  bool align(const sensor_msgs::msg::LaserScan &target_scan,
             const sensor_msgs::msg::LaserScan &source_scan, Sophus::SE2d &Tts,
             const size_t iterations = 10, const bool verbose = true,
             const SolverType type = SolverType::GN) {
    switch (type) {
    case SolverType::GN:
      return align_GN(target_scan, source_scan, Tts, iterations, verbose);
    case SolverType::G2o:
      return align_G2o(target_scan, source_scan, Tts, iterations, verbose);
    case SolverType::Ceres:
      return align_Ceres(target_scan, source_scan, Tts, iterations, verbose);
    }
    return false;
  }
  bool align_GN(const sensor_msgs::msg::LaserScan &target_scan,
                const sensor_msgs::msg::LaserScan &source_scan,
                Sophus::SE2d &Tts, const size_t iterations = 10,
                const bool verbose = true);
  bool align_G2o(const sensor_msgs::msg::LaserScan &target_scan,
                 const sensor_msgs::msg::LaserScan &source_scan,
                 Sophus::SE2d &Tts, const size_t iterations = 10,
                 const bool verbose = true);
  bool align_Ceres(const sensor_msgs::msg::LaserScan &target_scan,
                   const sensor_msgs::msg::LaserScan &source_scan,
                   Sophus::SE2d &Tts, const size_t iterations = 10,
                   const bool verbose = true);

  bool scan2map(const sensor_msgs::msg::LaserScan &source_scan,
                Sophus::SE2d &Tts, const size_t iterations = 10,
                const bool verbose = true,
                const SolverType type = SolverType::GN) {
    switch (type) {
    case SolverType::GN:
      return scan2map_GN(source_scan, Tts, iterations, verbose);
    case SolverType::G2o:
      return scan2map_G2o(source_scan, Tts, iterations, verbose);
    case SolverType::Ceres:
      return scan2map_Ceres(source_scan, Tts, iterations, verbose);
    }
    return false;
  }
  bool scan2map_G2o(const sensor_msgs::msg::LaserScan &source_scan,
                    Sophus::SE2d &Tts, const size_t iterations = 10,
                    const bool verbose = true);
  bool scan2map_Ceres(const sensor_msgs::msg::LaserScan &source_scan,
                      Sophus::SE2d &Tts, const size_t iterations = 10,
                      const bool verbose = true);
  bool scan2map_GN(const sensor_msgs::msg::LaserScan &source_scan,
                   Sophus::SE2d &Tts, const size_t iterations = 10,
                   const bool verbose = true);

  cv::Mat get_dist_map() const;
  bool get_dist_map(const std::string file_path) const;

  cv::Mat get_likelihood_field() const;
  bool get_likelihood_field(const std::string file_path) const;

  // https://en.wikipedia.org/wiki/Bilinear_interpolation
  // To get accurate value, 0.5 should be subtracted from the image coordinate
  // so that physical coordinate is consistent with distmap coordinate
  double get_value(double r, double c) const {
    r = std::max(0.0, r);
    r = std::min(double(dist_map_.rows - 1), r);
    c = std::max(0.0, c);
    c = std::min(double(dist_map_.cols - 1), c);

    const double dr = r - std::floor(r);
    const double dc = c - std::floor(c);
    return (1 - dc) * (1 - dr) * dist_map_.at<double>(r, c) +
           (1 - dc) * dr * dist_map_.at<double>(std::ceil(r), c) +
           dc * (1 - dr) * dist_map_.at<double>(r, std::ceil(c)) +
           dc * dr * dist_map_.at<double>(std::ceil(r), std::ceil(c));
  }

  int resolution() const { return resolution_; }
  int img_offset() const { return img_offset_; }
  int img_size() const { return img_size_; }
  const cv::Mat &data() { return dist_map_; }
  int cell_range() const { return local_patch_.cell_range; }

  bool outside(const double x, const double y, const double boarder = 0) const {
    return x < boarder || x >= dist_map_.cols - boarder || y < boarder ||
           y >= dist_map_.rows - boarder;
  }

  bool update_dist_map(const Eigen::Vector2i &img_coord);

private:
  LocalPatch local_patch_;
  int img_size_ = 1000;
  int img_offset_ = 500;
  int resolution_ = 20; // cell / m
  double max_cell_dist_ = 0;
  double sigma_ = 0.25;
  cv::Mat dist_map_;

  const int min_valid_ = 20;
  const double robust_kernel_delta = 0.8;
};