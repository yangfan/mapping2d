#pragma once

#include "LikelihoodField.h"
#include "OGM.h"

#include <Eigen/Core>

#include <vector>

class LikelihoodFieldMR {
public:
  LikelihoodFieldMR(const int img_sz, const double resolution)
      : base_img_size_(img_sz), base_resolution_(resolution) {
    likelihood_fields_.reserve(ratios_.size());

    for (const double ratio : ratios_) {
      likelihood_fields_.emplace_back(1 / ratio, base_img_size_ * ratio,
                                      base_resolution_ * ratio, 0.25 * ratio);
    }
  }

  bool set_dist_map(const OccupancyGridMap &ogm);
  bool set_dist_map(const cv::Mat &grid_map);

  cv::Mat dist_map(const size_t level) const {
    return likelihood_fields_[level].get_dist_map();
  }

  using SolverType = LikelihoodField::SolverType;
  bool scan2map(const sensor_msgs::msg::LaserScan &source_scan,
                Sophus::SE2d &Tts, const size_t iterations = 10,
                const bool verbose = true,
                const SolverType type = SolverType::G2o);

private:
  std::vector<double> ratios_ = {0.125, 0.25, 0.5, 1.0};
  std::vector<double> robust_kernel_deltas_ = {0.2, 0.3, 0.6, 0.8};
  std::vector<LikelihoodField> likelihood_fields_;

  int base_img_size_ = 1000;
  double base_resolution_ = 20;

  int min_inliers_ = 100;
  double min_inliers_ratio_ = 0.4;

  bool scan2map_lv_G2o(const size_t lv, const std::vector<double> &ranges,
                       const std::vector<double> angles, Sophus::SE2d &Tts,
                       const size_t iterations = 10, const bool verbose = true);

  bool scan2map_lv_Ceres(const size_t lv, const std::vector<double> &ranges,
                         const std::vector<double> angles, Sophus::SE2d &Tts,
                         const size_t iterations = 10,
                         const bool verbose = true);
};