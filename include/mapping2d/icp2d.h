#pragma once
#include "KDTree.hpp"
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sophus/se2.hpp>

namespace ICP2D {

using KDTree = KDTree<double, 2>;
using Point = KDTree::PointType;
using PointCloud = KDTree::PointCloud;

constexpr double kmax_dist = 0.1;
constexpr size_t kmin_valid = 20;
constexpr double khb_delta = 0.8;

enum class OptimizerType { G2o, Ceres, GN };

inline Point scan2point(const double range, const double angle) {
  return Point(range * std::cos(angle), range * std::sin(angle));
}

inline std::pair<double, double> point2scan(const Point &point) {
  return {point.norm(), std::atan2(point.y(), point.x())};
}

bool setKDTree(KDTree &kdtree, const sensor_msgs::msg::LaserScan &scan);

bool P2PGN(const sensor_msgs::msg::LaserScan &target,
           const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
           const size_t iterations = 10, bool verbose = true);

bool P2PG2o(const sensor_msgs::msg::LaserScan &target,
            const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
            const size_t iterations = 10, bool verbose = true);

bool P2PCeres(const sensor_msgs::msg::LaserScan &target,
              const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
              const size_t iterations = 10, bool verbose = true);

inline bool P2P(const sensor_msgs::msg::LaserScan &target,
                const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
                const size_t iterations = 10, bool verbose = true,
                OptimizerType opt_type = OptimizerType::GN) {
  switch (opt_type) {
  case OptimizerType::G2o:
    return P2PG2o(target, source, Tts, iterations, verbose);
  case OptimizerType::Ceres:
    return P2PCeres(target, source, Tts, iterations, verbose);
  case OptimizerType::GN:
    return P2PGN(target, source, Tts, iterations, verbose);
  default:
    LOG(ERROR) << "Please select one of the optimizer types: G2o, Ceres, GN.";
    return false;
  }
}
} // namespace ICP2D