#include "LikelihoodFieldMR.h"
#include "ceres_types.h"
#include "g2o_types.h"

#include <execution>

#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <ceres/ceres.h>

bool LikelihoodFieldMR::set_dist_map(const OccupancyGridMap &ogm) {
  return set_dist_map(ogm.map());
}

bool LikelihoodFieldMR::set_dist_map(const cv::Mat &grid_map) {
  for (auto &lf : likelihood_fields_) {
    lf.reset();
  }
  std::vector<int> idx(ratios_.size());
  std::iota(idx.begin(), idx.end(), 0);
  for (int r = 0; r < grid_map.rows; ++r) {
    for (int c = 0; c < grid_map.cols; ++c) {
      if (grid_map.at<uchar>(r, c) < 127) {
        std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                      [&c, &r, this](const int id) {
                        const int x = c * ratios_[id];
                        const int y = r * ratios_[id];
                        if (!likelihood_fields_[id].outside(
                                x, y, likelihood_fields_[id].cell_range())) {
                          likelihood_fields_[id].update_dist_map(
                              Eigen::Vector2i(x, y));
                        }
                      });
      }
    }
  }
  return true;
}

bool LikelihoodFieldMR::scan2map(const sensor_msgs::msg::LaserScan &source_scan,
                                 Sophus::SE2d &Tts, const size_t iterations,
                                 const bool verbose, const SolverType type) {
  if (source_scan.ranges.empty()) {
    return false;
  }

  std::vector<double> ranges;
  std::vector<double> angles;
  ranges.reserve(source_scan.ranges.size());
  angles.reserve(source_scan.ranges.size());
  for (size_t sid = 0; sid < source_scan.ranges.size(); ++sid) {
    const double range = source_scan.ranges[sid];
    const double angle =
        source_scan.angle_min + sid * source_scan.angle_increment;
    if (range < source_scan.range_min || range > source_scan.range_max ||
        angle < source_scan.angle_min + 30 * M_PI / 180 ||
        angle > source_scan.angle_max - 30 * M_PI / 180) {
      continue;
    }
    if (range > 15.0) {
      continue;
    }
    ranges.emplace_back(range);
    angles.emplace_back(angle);
  }

  for (size_t level = 0; level < ratios_.size(); ++level) {
    bool valid_match = false;
    switch (type) {
    case SolverType::G2o:
      valid_match =
          scan2map_lv_G2o(level, ranges, angles, Tts, iterations, verbose);
      break;
    case SolverType::Ceres:
      valid_match =
          scan2map_lv_Ceres(level, ranges, angles, Tts, iterations, verbose);
      break;
    default:
      valid_match =
          scan2map_lv_G2o(level, ranges, angles, Tts, iterations, verbose);
    }
    if (!valid_match) {
      return false;
    }
  }
  return true;
}

bool LikelihoodFieldMR::scan2map_lv_G2o(const size_t lv,
                                        const std::vector<double> &ranges,
                                        const std::vector<double> angles,
                                        Sophus::SE2d &Tts,
                                        const size_t iterations,
                                        const bool verbose) {

  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;
  auto *solver = new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  VertexSE2 *vertex = new VertexSE2();
  vertex->setEstimate(Tts);
  vertex->setId(0);
  optimizer.addVertex(vertex);

  std::vector<EdgeLikelihood *> edges;
  edges.reserve(ranges.size());
  for (size_t sid = 0; sid < ranges.size(); ++sid) {
    const double range = ranges[sid];
    const double angle = angles[sid];
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_pt = Tts * query_pt_s;

    const Eigen::Vector2d query_coord =
        likelihood_fields_[lv].resolution() * query_pt +
        likelihood_fields_[lv].img_offset() * Eigen::Vector2d::Ones();
    if (likelihood_fields_[lv].outside(query_coord.x(), query_coord.y(),
                                       likelihood_fields_[lv].cell_range())) {
      continue;
    }

    EdgeLikelihood *edge =
        new EdgeLikelihood(&likelihood_fields_[lv], range, angle);
    edge->setMeasurement(query_pt_s);
    edge->setVertex(0, vertex);
    edge->setId(edges.size());
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
    rk->setDelta(robust_kernel_deltas_[lv]);
    edge->setRobustKernel(rk);
    optimizer.addEdge(edge);

    edges.emplace_back(edge);
  }
  optimizer.setVerbose(verbose);
  optimizer.initializeOptimization();
  optimizer.optimize(iterations);

  Tts = vertex->estimate();

  int num_inliers = std::accumulate(
      edges.begin(), edges.end(), 0,
      [&lv, this](int cnt, const EdgeLikelihood *edge) {
        if (edge->level() == 0 && edge->chi2() < robust_kernel_deltas_[lv]) {
          return cnt + 1;
        }
        return cnt;
      });

  const double inliers_ratio = double(num_inliers) / edges.size();

  return num_inliers > min_inliers_ && inliers_ratio > min_inliers_ratio_;
}

bool LikelihoodFieldMR::scan2map_lv_Ceres(const size_t lv,
                                          const std::vector<double> &ranges,
                                          const std::vector<double> angles,
                                          Sophus::SE2d &Tts,
                                          const size_t iterations,
                                          const bool verbose) {
  double pose[3] = {Tts.translation().x(), Tts.translation().y(),
                    Tts.so2().log()};
  ceres::Problem problem;
  std::vector<ceres::ResidualBlockId> edges;
  edges.reserve(ranges.size());
  for (size_t sid = 0; sid < ranges.size(); ++sid) {
    const double range = ranges[sid];
    const double angle = angles[sid];
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_pt = Tts * query_pt_s;
    const Eigen::Vector2d query_coord =
        likelihood_fields_[lv].resolution() * (query_pt) +
        likelihood_fields_[lv].img_offset() * Eigen::Vector2d::Ones();
    if (likelihood_fields_[lv].outside(query_coord.x(), query_coord.y(),
                                       likelihood_fields_[lv].cell_range())) {
      continue;
    }
    ceres::CostFunction *cost_func = new LikelihoodAlignment(
        &likelihood_fields_[lv], query_pt_s, range, angle);
    ceres::LossFunction *loss_func =
        new ceres::HuberLoss(robust_kernel_deltas_[lv]);
    auto block_id = problem.AddResidualBlock(cost_func, loss_func, pose);
    edges.emplace_back(block_id);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = verbose;
  // options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.num_threads = 4;
  options.max_linear_solver_iterations = iterations;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Tts = Sophus::SE2d(pose[2], Eigen::Vector2d(pose[0], pose[1]));

  int num_inliers = std::accumulate(
      edges.begin(), edges.end(), 0,
      [&lv, &problem, this](int cnt, const ceres::ResidualBlockId &edge) {
        auto *likelihood_cost = static_cast<const LikelihoodAlignment *>(
            problem.GetCostFunctionForResidualBlock(edge));
        double cost = 0;
        problem.EvaluateResidualBlockAssumingParametersUnchanged(
            edge, false, &cost, nullptr, nullptr);

        if (!likelihood_cost->is_outlier() &&
            4 * cost < robust_kernel_deltas_[lv]) {
          return cnt + 1;
        }
        return cnt;
      });

  const double inliers_ratio = double(num_inliers) / edges.size();

  return num_inliers > min_inliers_ && inliers_ratio > min_inliers_ratio_;
}