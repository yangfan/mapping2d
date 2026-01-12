#include "LikelihoodField.h"
#include "ceres_types.h"
#include "g2o_types.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <cmath>

bool LikelihoodField::set_dist_map(
    const sensor_msgs::msg::LaserScan &target_scan) {
  if (target_scan.ranges.empty()) {
    return false;
  }
  reset();
  for (size_t tid = 0; tid < target_scan.ranges.size(); ++tid) {
    const double range = target_scan.ranges[tid];
    const double angle =
        target_scan.angle_min + tid * target_scan.angle_increment;
    if (range < target_scan.range_min || range > target_scan.range_max ||
        angle < target_scan.angle_min + 30 * M_PI / 180 ||
        angle > target_scan.angle_max - 30 * M_PI / 180) {
      continue;
    }

    const Eigen::Vector2i img_coord(
        range * std::cos(angle) * resolution_ + img_offset_,
        range * std::sin(angle) * resolution_ + img_offset_);
    update_dist_map(img_coord);
  }

  return true;
}

bool LikelihoodField::set_dist_map(const OccupancyGridMap &grid_map) {
  if (grid_map.map().empty()) {
    return false;
  }
  reset();
  for (int r = 0; r < grid_map.map().rows; ++r) {
    for (int c = 0; c < grid_map.map().cols; ++c) {
      if (grid_map.map().at<uchar>(r, c) < 127 &&
          !outside(c, r, local_patch_.cell_range) &&
          dist_map_.at<double>(r, c) > 0.0) {
        update_dist_map(Eigen::Vector2i(c, r));
      }
    }
  }
  return true;
}

bool LikelihoodField::update_dist_map(const Eigen::Vector2i &img_coord) {
  if (outside(img_coord.x(), img_coord.y())) {
    return false;
  }
  for (int delta_x = -local_patch_.cell_range;
       delta_x <= local_patch_.cell_range; ++delta_x) {
    for (int delta_y = -local_patch_.cell_range;
         delta_y <= local_patch_.cell_range; ++delta_y) {
      const double cell_coord_x = img_coord.x() + delta_x;
      const double cell_coord_y = img_coord.y() + delta_y;
      if (outside(cell_coord_x, cell_coord_y)) {
        continue;
      }
      const double cell_value = local_patch_.cell_value(delta_x, delta_y);
      dist_map_.at<double>(cell_coord_y, cell_coord_x) = std::min(
          dist_map_.at<double>(cell_coord_y, cell_coord_x), cell_value);
    }
  }

  return true;
}

cv::Mat LikelihoodField::get_dist_map() const {
  cv::Mat dist_img(dist_map_.rows, dist_map_.cols, CV_8UC1);
  for (int r = 0; r < dist_map_.rows; ++r) {
    for (int c = 0; c < dist_map_.cols; ++c) {
      dist_img.at<uchar>(r, c) =
          uchar(dist_map_.at<double>(r, c) * 255 / max_cell_dist_);
    }
  }
  return dist_img;
}
bool LikelihoodField::get_dist_map(const std::string file_path) const {
  return cv::imwrite(file_path, get_dist_map());
}

cv::Mat LikelihoodField::get_likelihood_field() const {
  const double sigma_cell = sigma_ * resolution_;
  cv::Mat likelihood_field(img_size_, img_size_, CV_8UC1);
  for (int r = 0; r < dist_map_.rows; ++r) {
    for (int c = 0; c < dist_map_.cols; ++c) {
      const double dist = dist_map_.at<double>(r, c);
      likelihood_field.at<uchar>(r, c) = uchar(
          255 * std::exp(-0.5 * (dist / sigma_cell) * (dist / sigma_cell)));
    }
  }
  return likelihood_field;
}
bool LikelihoodField::get_likelihood_field(const std::string file_path) const {
  return cv::imwrite(file_path, get_likelihood_field());
}

bool LikelihoodField::align_GN(const sensor_msgs::msg::LaserScan &target_scan,
                               const sensor_msgs::msg::LaserScan &source_scan,
                               Sophus::SE2d &Tts, const size_t iterations,
                               const bool verbose) {
  if (target_scan.ranges.empty() || source_scan.ranges.empty()) {
    LOG(ERROR) << "Empty laser scan.";
    return false;
  }
  if (!set_dist_map(target_scan)) {
    LOG(ERROR) << "Failed to set distance map.";
    return false;
  }
  Sophus::SE2d pose = Tts;

  double last_error = std::numeric_limits<double>::max();

  for (size_t i = 0; i < iterations; ++i) {
    double curr_err = 0;
    int valid_cnt = 0;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (size_t sid = 0; sid < source_scan.ranges.size(); ++sid) {
      const double range = source_scan.ranges[sid];
      const double angle =
          source_scan.angle_min + sid * source_scan.angle_increment;
      if (range < source_scan.range_min || range > source_scan.range_max ||
          angle < source_scan.angle_min + 30 * M_PI / 180 ||
          angle > source_scan.angle_max - 30 * M_PI / 180) {
        continue;
      }
      Eigen::Vector2d query_pt =
          pose *
          Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
      Eigen::Vector2d query_coord =
          resolution_ * query_pt + img_offset_ * Eigen::Vector2d::Ones();

      if (outside(query_coord.x(), query_coord.y(), local_patch_.cell_range)) {
        continue;
      }

      double dist = get_value(query_coord.y(), query_coord.x());
      if (dist >= max_cell_dist_) {
        continue;
      }
      const double grad_dist_x =
          0.5 * (get_value(query_coord.y(), query_coord.x() + 1) -
                 get_value(query_coord.y(), query_coord.x() - 1));
      const double grad_dist_y =
          0.5 * (get_value(query_coord.y() + 1, query_coord.x()) -
                 get_value(query_coord.y() - 1, query_coord.x()));

      const double theta = pose.so2().log();
      const Eigen::Matrix<double, 1, 3> jacobian(
          resolution_ * grad_dist_x, resolution_ * grad_dist_y,
          resolution_ * range *
              (-grad_dist_x * std::sin(angle + theta) +
               grad_dist_y * std::cos(angle + theta)));

      H += jacobian.transpose() * jacobian;
      b += -jacobian.transpose() * dist;
      valid_cnt++;
      curr_err += dist * dist;
    }
    if (valid_cnt < min_valid_) {
      return false;
    }
    const double avg_err = curr_err / valid_cnt;
    if (avg_err > last_error) {
      break;
    }
    last_error = avg_err;
    const Eigen::Vector3d delta = H.ldlt().solve(b);
    if (std::isnan(delta[0])) {
      break;
    }
    pose.translation() += delta.head<2>();
    pose.so2() = pose.so2() * Sophus::SO2d::exp(delta[2]);
    if (verbose) {
      LOG(INFO) << "Iter " << i << " avg err : " << avg_err;
    }
  }
  Tts = pose;

  return true;
}

bool LikelihoodField::align_G2o(const sensor_msgs::msg::LaserScan &target_scan,
                                const sensor_msgs::msg::LaserScan &source_scan,
                                Sophus::SE2d &Tts, const size_t iterations,
                                const bool verbose) {
  if (target_scan.ranges.empty() || source_scan.ranges.empty()) {
    return false;
  }
  if (!set_dist_map(target_scan)) {
    return false;
  }
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

  for (size_t sid = 0; sid < source_scan.ranges.size(); ++sid) {
    const double range = source_scan.ranges[sid];
    const double angle =
        source_scan.angle_min + sid * source_scan.angle_increment;
    if (range < source_scan.range_min || range > source_scan.range_max ||
        angle < source_scan.angle_min + 30 * M_PI / 180 ||
        angle > source_scan.angle_max - 30 * M_PI / 180) {
      continue;
    }
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_pt = Tts * query_pt_s;
    const Eigen::Vector2d query_coord =
        resolution_ * query_pt + img_offset_ * Eigen::Vector2d::Ones();
    if (outside(query_coord.x(), query_coord.y(), local_patch_.cell_range)) {
      continue;
    }

    EdgeLikelihood *edge = new EdgeLikelihood(this, range, angle);
    edge->setMeasurement(query_pt_s);
    edge->setVertex(0, vertex);
    edge->setId(0);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
    rk->setDelta(robust_kernel_delta);
    edge->setRobustKernel(rk);
    optimizer.addEdge(edge);
  }
  optimizer.setVerbose(verbose);
  optimizer.initializeOptimization();
  optimizer.optimize(iterations);

  Tts = vertex->estimate();

  return true;
}

bool LikelihoodField::align_Ceres(
    const sensor_msgs::msg::LaserScan &target_scan,
    const sensor_msgs::msg::LaserScan &source_scan, Sophus::SE2d &Tts,
    const size_t iterations, const bool verbose) {
  if (target_scan.ranges.empty() || source_scan.ranges.empty()) {
    return false;
  }
  if (!set_dist_map(target_scan)) {
    return false;
  }

  double pose[3] = {Tts.translation().x(), Tts.translation().y(),
                    Tts.so2().log()};
  ceres::Problem problem;
  for (size_t sid = 0; sid < source_scan.ranges.size(); ++sid) {
    const double range = source_scan.ranges[sid];
    const double angle =
        source_scan.angle_min + sid * source_scan.angle_increment;
    if (range < source_scan.range_min || range > source_scan.range_max ||
        angle < source_scan.angle_min + 30 * M_PI / 180 ||
        angle > source_scan.angle_max - 30 * M_PI / 180) {
      continue;
    }
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_coord = resolution_ * (Tts * query_pt_s) +
                                        img_offset_ * Eigen::Vector2d::Ones();
    if (outside(query_coord.x(), query_coord.y())) {
      continue;
    }
    ceres::CostFunction *cost_func =
        new LikelihoodAlignment(this, query_pt_s, range, angle);
    ceres::LossFunction *loss_func = new ceres::HuberLoss(robust_kernel_delta);
    problem.AddResidualBlock(cost_func, loss_func, pose);
  }
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = verbose;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.max_linear_solver_iterations = iterations;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (verbose) {
    LOG(INFO) << summary.FullReport();
  }

  Tts = Sophus::SE2d(pose[2], Eigen::Vector2d(pose[0], pose[1]));

  return true;
}

void LikelihoodField::reset() {
  dist_map_.release();
  dist_map_ = cv::Mat(img_size_, img_size_, CV_64FC1, max_cell_dist_);
}

bool LikelihoodField::scan2map_G2o(
    const sensor_msgs::msg::LaserScan &source_scan, Sophus::SE2d &Tts,
    const size_t iterations, const bool verbose) {
  if (source_scan.ranges.empty()) {
    return false;
  }

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

  int cnt = 0;

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
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_pt = Tts * query_pt_s;
    const Eigen::Vector2d query_coord =
        resolution_ * query_pt + img_offset_ * Eigen::Vector2d::Ones();
    if (outside(query_coord.x(), query_coord.y(), local_patch_.cell_range)) {
      continue;
    }

    EdgeLikelihood *edge = new EdgeLikelihood(this, range, angle);
    edge->setMeasurement(query_pt_s);
    edge->setVertex(0, vertex);
    edge->setId(cnt++);
    edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber();
    rk->setDelta(robust_kernel_delta);
    edge->setRobustKernel(rk);
    optimizer.addEdge(edge);
  }
  optimizer.setVerbose(verbose);
  optimizer.initializeOptimization();
  optimizer.optimize(iterations);

  Tts = vertex->estimate();

  return true;
}

bool LikelihoodField::scan2map_Ceres(
    const sensor_msgs::msg::LaserScan &source_scan, Sophus::SE2d &Tts,
    const size_t iterations, const bool verbose) {
  if (source_scan.ranges.empty()) {
    return false;
  }

  double pose[3] = {Tts.translation().x(), Tts.translation().y(),
                    Tts.so2().log()};
  ceres::Problem problem;

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
    const Eigen::Vector2d query_pt_s =
        Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
    const Eigen::Vector2d query_pt = Tts * query_pt_s;
    const Eigen::Vector2d query_coord =
        resolution_ * (query_pt) + img_offset_ * Eigen::Vector2d::Ones();
    if (outside(query_coord.x(), query_coord.y(), local_patch_.cell_range)) {
      continue;
    }
    ceres::CostFunction *cost_func =
        new LikelihoodAlignment(this, query_pt_s, range, angle);
    ceres::LossFunction *loss_func = new ceres::HuberLoss(robust_kernel_delta);
    problem.AddResidualBlock(cost_func, loss_func, pose);
  }
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = verbose;
  options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.max_linear_solver_iterations = iterations;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (verbose) {
    LOG(INFO) << summary.FullReport();
  }

  Tts = Sophus::SE2d(pose[2], Eigen::Vector2d(pose[0], pose[1]));

  return true;
}

bool LikelihoodField::scan2map_GN(
    const sensor_msgs::msg::LaserScan &source_scan, Sophus::SE2d &Tts,
    const size_t iterations, const bool verbose) {
  if (source_scan.ranges.empty()) {
    return false;
  }
  Sophus::SE2d pose = Tts;

  double last_error = std::numeric_limits<double>::max();

  for (size_t i = 0; i < iterations; ++i) {
    double curr_err = 0;
    int valid_cnt = 0;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
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
      Eigen::Vector2d query_pt =
          pose *
          Eigen::Vector2d(range * std::cos(angle), range * std::sin(angle));
      Eigen::Vector2d query_coord =
          resolution_ * query_pt + img_offset_ * Eigen::Vector2d::Ones();

      // if (outside(query_coord.x(), query_coord.y(), local_patch_.cell_range))
      // {
      if (outside(query_coord.x(), query_coord.y(), 10)) {
        continue;
      }

      // to get dist value
      query_coord = query_coord - Eigen::Vector2d(0.5, 0.5);
      double dist = get_value(query_coord.y(), query_coord.x());
      // if (dist >= max_cell_dist_) {
      //   continue;
      // }
      const double grad_dist_x =
          0.5 * (get_value(query_coord.y(), query_coord.x() + 1) -
                 get_value(query_coord.y(), query_coord.x() - 1));
      const double grad_dist_y =
          0.5 * (get_value(query_coord.y() + 1, query_coord.x()) -
                 get_value(query_coord.y() - 1, query_coord.x()));

      const double theta = pose.so2().log();
      const Eigen::Matrix<double, 1, 3> jacobian(
          resolution_ * grad_dist_x, resolution_ * grad_dist_y,
          resolution_ * range *
              (-grad_dist_x * std::sin(angle + theta) +
               grad_dist_y * std::cos(angle + theta)));

      H += jacobian.transpose() * jacobian;
      b += -jacobian.transpose() * dist;
      valid_cnt++;
      curr_err += dist * dist;
    }
    if (valid_cnt < min_valid_) {
      return false;
    }
    const double avg_err = curr_err / valid_cnt;
    if (avg_err > last_error) {
      break;
    }
    last_error = avg_err;
    const Eigen::Vector3d delta = H.ldlt().solve(b);
    if (std::isnan(delta[0])) {
      break;
    }
    pose.translation() += delta.head<2>();
    pose.so2() = pose.so2() * Sophus::SO2d::exp(delta[2]);
    if (verbose) {
      LOG(INFO) << "Iter " << i << " avg err : " << avg_err;
    }
  }
  Tts = pose;

  return true;
}
