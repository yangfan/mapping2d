#include "icp2d.h"
#include "ceres_types.h"
#include "g2o_types.h"

#include <memory>

#include <Eigen/Core>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace ICP2D {

bool setKDTree(KDTree &kdtree, const sensor_msgs::msg::LaserScan &scan) {
  if (scan.ranges.empty()) {
    return false;
  }
  KDTree::PointCloud cloud;
  cloud.reserve(scan.ranges.size());
  for (size_t pid = 0; pid < scan.ranges.size(); ++pid) {
    if (scan.ranges[pid] < scan.range_min ||
        scan.ranges[pid] > scan.range_max) {
      continue;
    }
    cloud.emplace_back(scan2point(scan.ranges[pid],
                                  scan.angle_min + pid * scan.angle_increment));
  }
  kdtree.setInputCloud(std::move(cloud));

  return true;
}

bool P2PGN(const sensor_msgs::msg::LaserScan &target,
           const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
           const size_t iterations, bool verbose) {
  if (target.ranges.empty() || source.ranges.empty()) {
    return false;
  }
  Sophus::SE2d pose = Tts;
  KDTree kdtree;
  if (!setKDTree(kdtree, target)) {
    return false;
  }

  auto jacobian = [](const double range, const double angle,
                     const Sophus::SE2d &pose) {
    Eigen::Matrix<double, 2, 3> jac;
    jac << 1, 0, -range * std::sin(angle + pose.so2().log()), 0, 1,
        range * std::cos(angle + pose.so2().log());
    return jac;
  };

  double last_err = std::numeric_limits<double>::max();
  for (size_t i = 0; i < iterations; ++i) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    size_t valid_cnt = 0;
    double curr_err2 = 0;

    for (size_t sid = 0; sid < source.ranges.size(); ++sid) {
      const double range = source.ranges[sid];
      const double angle = source.angle_min + sid * source.angle_increment;
      if (range < source.range_min || range > source.range_max ||
          angle < source.angle_min + 30 * M_PI / 180.0 ||
          angle > source.angle_max - 30 * M_PI / 180.0) {

        continue;
      }

      const Point query_pt = pose * scan2point(range, angle);

      std::vector<int> tidx;
      std::vector<double> tdist;
      if (!kdtree.nearest_neighbors(query_pt, 1, tidx, tdist)) {
        continue;
      }
      if (tdist[0] > kmax_dist) {
        continue;
      }

      const Eigen::Vector2d err = query_pt - kdtree.get_point(tidx[0]);
      const Eigen::Matrix<double, 2, 3> J = jacobian(range, angle, pose);
      H += J.transpose() * J;
      b += -J.transpose() * err;
      valid_cnt++;
      curr_err2 += err.squaredNorm();
    }
    if (valid_cnt < kmin_valid) {
      return false;
    }
    const double avg_err = curr_err2 / valid_cnt;
    const Eigen::Vector3d delta = H.ldlt().solve(b);
    if (avg_err > last_err || std::isnan(delta[0])) {
      break;
    }
    if (verbose) {
      LOG(INFO) << "Iter " << i << " err : " << avg_err;
    }
    last_err = avg_err;
    pose.translation() += delta.head<2>();
    pose.so2() = pose.so2() * Sophus::SO2d::exp(delta[2]);
  }
  Tts = pose;
  return true;
}

bool P2PG2o(const sensor_msgs::msg::LaserScan &target,
            const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
            const size_t iterations, bool verbose) {
  if (target.ranges.empty() || source.ranges.empty()) {
    return false;
  }
  KDTree kdtree;
  if (!setKDTree(kdtree, target)) {
    return false;
  }
  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 2>>;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

  auto *solver = new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);
  optimizer.setVerbose(verbose);

  VertexSE2 *vertex = new VertexSE2();
  vertex->setId(0);
  vertex->setEstimate(Tts);
  optimizer.addVertex(vertex);

  size_t edge_cnt = 0;
  for (size_t sid = 0; sid < source.ranges.size(); ++sid) {
    const double range = source.ranges[sid];
    const double angle = source.angle_min + sid * source.angle_increment;
    if (range < source.range_min || range > source.range_max ||
        angle < source.angle_min + 30 * M_PI / 180.0 ||
        angle > source.angle_max - 30 * M_PI / 180.0) {

      continue;
    }
    EdgeScan2D *edge = new EdgeScan2D(&kdtree);
    edge->setVertex(0, vertex);
    edge->setMeasurement(Eigen::Vector2d(range, angle));
    edge->setInformation(Eigen::Matrix2d::Identity());
    edge->setId(edge_cnt++);
    auto *rk_huber = new g2o::RobustKernelHuber();
    rk_huber->setDelta(khb_delta);
    edge->setRobustKernel(rk_huber);
    optimizer.addEdge(edge);
  }
  optimizer.initializeOptimization();
  optimizer.optimize(iterations);

  Tts = vertex->estimate();

  return true;
}

bool P2PCeres(const sensor_msgs::msg::LaserScan &target,
              const sensor_msgs::msg::LaserScan &source, Sophus::SE2d &Tts,
              const size_t iterations, bool verbose) {

  if (target.ranges.empty() || source.ranges.empty()) {
    return false;
  }
  KDTree kdtree;
  if (!setKDTree(kdtree, target)) {
    return false;
  }
  ceres::Problem problem;
  double pose[3] = {Tts.translation().x(), Tts.translation().y(),
                    Tts.so2().log()};
  for (size_t sid = 0; sid < source.ranges.size(); ++sid) {
    const double range = source.ranges[sid];
    const double angle = source.angle_min + sid * source.angle_increment;
    if (range < source.range_min || range > source.range_max ||
        angle < source.angle_min + 30 * M_PI / 180.0 ||
        angle > source.angle_max - 30 * M_PI / 180.0) {
      continue;
    }
    ceres::CostFunction *cost_func = new ICP2DCeres(range, angle, &kdtree);
    ceres::LossFunction *loss_func = new ceres::HuberLoss(khb_delta);
    problem.AddResidualBlock(cost_func, loss_func, pose);
  }

  ceres::Solver::Options options;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = verbose;
  options.max_num_iterations = iterations;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (verbose) {
    LOG(INFO) << summary.FullReport();
  }
  Tts = Sophus::SE2d(pose[2], Eigen::Vector2d(pose[0], pose[1]));

  return true;
}

} // namespace ICP2D