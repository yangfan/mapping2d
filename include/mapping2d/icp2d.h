#pragma once
#include "KDTree.hpp"
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sophus/se2.hpp>

namespace ICP2D {

using KDTree = KDTree<double, 2>;
using Point = KDTree::PointType;
using PointCloud = KDTree::PointCloud;

constexpr double kmax_p2p_dist2 = 0.1;
constexpr size_t kmin_valid = 20;
constexpr double khb_delta = 0.8;
constexpr int kmax_p2l_num = 5;

enum class OptimizerType { G2o, G2oMT, Ceres, CeresMT, GN };

inline Point scan2point(const double range, const double angle) {
  return Point(range * std::cos(angle), range * std::sin(angle));
}

inline std::pair<double, double> point2scan(const Point &point) {
  return {point.norm(), std::atan2(point.y(), point.x())};
}

inline bool line_fitting(const std::vector<Point> &points,
                         Eigen::Vector3d &line_coeffs) {
  if (points.size() < 3) {
    return false;
  }
  Eigen::MatrixX<double> A(points.size(), 3);
  A.setOnes();
  for (size_t i = 0; i < points.size(); ++i) {
    A.row(i).head<2>() = points[i].transpose();
  }
  Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
  line_coeffs = svd.matrixV().col(2);
  return true;
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
  return false;
}

bool P2LGN(const sensor_msgs::msg::LaserScan &target,
           const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
           const size_t iterations = 10, bool verbose = true);

bool P2LG2o(const sensor_msgs::msg::LaserScan &target,
            const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
            const size_t iterations = 10, bool verbose = true);

bool P2LG2oMT(const sensor_msgs::msg::LaserScan &target,
              const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
              const size_t iterations = 10, bool verbose = true);

bool P2LCeres(const sensor_msgs::msg::LaserScan &target,
              const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
              const size_t iterations = 10, bool verbose = true);

bool P2LCeresMT(const sensor_msgs::msg::LaserScan &target,
                const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
                const size_t iterations = 10, bool verbose = true);

inline bool P2L(const sensor_msgs::msg::LaserScan &target,
                const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
                const size_t iterations = 10, bool verbose = true,
                OptimizerType opt_type = OptimizerType::GN) {
  switch (opt_type) {
  case OptimizerType::G2o:
    return P2LG2o(target, source, Tts, iterations, verbose);
  case OptimizerType::G2oMT:
    return P2LG2oMT(target, source, Tts, iterations, verbose);
  case OptimizerType::Ceres:
    return P2LCeres(target, source, Tts, iterations, verbose);
  case OptimizerType::CeresMT:
    return P2LCeresMT(target, source, Tts, iterations, verbose);
  case OptimizerType::GN:
    return P2LGN(target, source, Tts, iterations, verbose);
  default:
    LOG(ERROR) << "Please select one of the optimizer types: G2o, G2oMT, "
                  "Ceres, CeresMT, GN.";
    return false;
  }
  return false;
}
} // namespace ICP2D