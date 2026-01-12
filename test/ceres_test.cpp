#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ceres_types.h"

#include <cmath>
#include <random>

TEST(Ceres, Data) {
  EXPECT_TRUE(true);
  const double angle = M_PI / 2;
  Sophus::SO2d rot(angle);
  LOG(INFO) << rot.log();
  double data[] = {std::cos(angle), std::sin(angle)};
  Eigen::Map<Sophus::SO2d> rot2(data);
  LOG(INFO) << rot2.log();

  Eigen::Quaterniond quat(1, 2, 3, 4);
  Sophus::SO3d rot_so3(quat);
  LOG(INFO) << rot_so3.log().transpose();
  double data_so3[] = {quat.x(), quat.y(), quat.z(), quat.w()};
  Eigen::Map<Sophus::SO3d> map_so3(data_so3);
  LOG(INFO) << map_so3.log().transpose();

  const Eigen::Vector2d pos(1, 2);
  Sophus::SE2d pose_2d(angle, pos);
  LOG(INFO) << pose_2d.log().transpose();
  double data_se2[] = {std::cos(angle), std::sin(angle), pos.x(), pos.y()};
  Eigen::Map<Sophus::SE2d> map_se2(data_se2);
  LOG(INFO) << map_se2.log().transpose();

  const Eigen::Vector3d pos3d(1, 2, 3);
  Sophus::SE3d pose_3d(quat, pos3d);
  LOG(INFO) << pose_3d.log().transpose();
  double data_se3[] = {quat.x(),  quat.y(),  quat.z(), quat.w(),
                       pos3d.x(), pos3d.y(), pos3d.z()};
  Eigen::Map<Sophus::SE3d> map_se3(data_se3);
  LOG(INFO) << map_se3.log().transpose();

  double ang[] = {2, 1};
  Eigen::Map<Sophus::SO2d> ang2(ang);
  LOG(INFO) << ang2.log();
  LOG(INFO) << std::atan2(1, 2);
}

struct SO3Cost {
  SO3Cost(const Sophus::SO3d &m) : measurement(m) {}
  template <typename T> bool operator()(const T *const x, T *residual) const {
    Eigen::Map<const Sophus::SO3<T>> rot3d(x);
    Eigen::Map<typename Sophus::SO3<T>::Tangent> err(residual);
    err = (measurement.inverse().template cast<T>() * rot3d).log();
    return true;
  }

  Sophus::SO3d measurement;
};

TEST(Ceres, SO3) {
  Eigen::Quaterniond quat(1, 2, 3, 4);
  Sophus::SO3d rot_so3(quat);

  Sophus::SO3d initial;

  ceres::Problem problem;
  Sophus::Manifold<Sophus::SO3> *manu = new Sophus::Manifold<Sophus::SO3>;
  problem.AddParameterBlock(initial.data(), 4, manu);

  ceres::CostFunction *fuc =
      new ceres::AutoDiffCostFunction<SO3Cost, 3, 4>(new SO3Cost(rot_so3));
  problem.AddResidualBlock(fuc, nullptr, initial.data());

  ceres::Solver::Options opts;
  opts.max_num_iterations = 200;
  ceres::Solver::Summary sum;

  ceres::Solve(opts, &problem, &sum);

  Sophus::SO3d::Tangent diff = (initial * rot_so3.inverse()).log();
  LOG(INFO) << "diff: " << diff.transpose();
  EXPECT_LT(diff.norm(), 1e-5);
}

struct SE2Cost {
  SE2Cost(const Sophus::SE2d m) : measurement(m) {}
  template <typename T>
  bool operator()(const T *const quat, T *residual) const {
    Eigen::Map<const Sophus::SE2<T>> pose(quat);
    Eigen::Map<typename Sophus::SE2<T>::Tangent> err(residual);
    err = (measurement.inverse().template cast<T>() * pose).log();

    return true;
  }

  Sophus::SE2d measurement;
};

TEST(Ceres, SE2) {
  Sophus::SE2d real_pose(M_PI / 3, Eigen::Vector2d(1, 2));
  Sophus::SE2d optimized_pose;

  ceres::Problem problem;

  auto mani = new Sophus::Manifold<Sophus::SE2>;
  problem.AddParameterBlock(optimized_pose.data(), 4, mani);

  auto cost_func =
      new ceres::AutoDiffCostFunction<SE2Cost, 3, 4>(new SE2Cost(real_pose));
  problem.AddResidualBlock(cost_func, nullptr, optimized_pose.data());

  ceres::Solver::Options opts;
  opts.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  Sophus::SE2d::Tangent diff = (real_pose.inverse() * optimized_pose).log();
  LOG(INFO) << "diff: " << diff.transpose();
  EXPECT_LT(diff.norm(), 1e-5);
}

TEST(Ceres, SE2EdgeCost) {
  Sophus::SE2d real_v1(M_PI / 2, Eigen::Vector2d(1, 2));
  Sophus::SE2d real_v2(M_PI / 4, Eigen::Vector2d(3, 4));
  Sophus::SE2d real_T12 = real_v1.inverse() * real_v2;

  ceres::Problem problem;

  const Sophus::SE2d noise(10 * M_PI / 180, Eigen::Vector2d(0.3, 0.4));
  Sophus::SE2d v1_estimate = real_v1 * noise;
  Sophus::SE2d v2_estimate = real_v2 * noise;

  auto manifold = new Sophus::Manifold<Sophus::SE2>;
  problem.AddParameterBlock(v1_estimate.data(), 4, manifold);
  problem.AddParameterBlock(v2_estimate.data(), 4, manifold);

  auto cost_func = new ceres::AutoDiffCostFunction<SE2EdgeCost, 3, 4, 4>(
      new SE2EdgeCost(real_T12, Eigen::Matrix3d::Identity()));
  auto residual_id = problem.AddResidualBlock(
      cost_func, nullptr, v1_estimate.data(), v2_estimate.data());
  double cost = 0;

  Sophus::SE2d::Tangent residual;
  // problem.EvaluateResidualBlock(residual_id, false, &cost, residual.data(),
  //                               nullptr);
  problem.EvaluateResidualBlockAssumingParametersUnchanged(
      residual_id, false, &cost, residual.data(), nullptr);
  LOG(INFO) << "initial cost: " << cost
            << ", residual: " << residual.transpose();
  Sophus::SE2d::Tangent tan =
      (v1_estimate.inverse() * v2_estimate * real_T12.inverse()).log();
  LOG(INFO) << "compute res: " << tan.transpose() << ", norm: " << tan.norm()
            << ", 0.5 sq norm: " << 0.5 * tan.squaredNorm();

  ceres::Solver::Options opts;
  opts.min_linear_solver_iterations = 100;
  opts.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  ceres::Solve(opts, &problem, &summary);
  Sophus::SE2d::Tangent diff =
      (v1_estimate.inverse() * v2_estimate * real_T12.inverse()).log();
  LOG(INFO) << "diff: " << diff.transpose();
  EXPECT_LT(diff.norm(), 1e-5);

  // problem.EvaluateResidualBlock(residual_id, false, &cost, residual.data(),
  //                               nullptr);
  problem.EvaluateResidualBlockAssumingParametersUnchanged(
      residual_id, false, &cost, residual.data(), nullptr);
  LOG(INFO) << "after cost: " << cost << ", residual: " << residual.transpose();
  tan = (v1_estimate.inverse() * v2_estimate * real_T12.inverse()).log();
  LOG(INFO) << "compute res: " << tan.transpose() << ", norm: " << tan.norm()
            << ", 0.5 sq norm: " << 0.5 * tan.squaredNorm();
  double ct = 0;
  problem.EvaluateResidualBlockAssumingParametersUnchanged(
      residual_id, false, &ct, nullptr, nullptr);
  LOG(INFO) << "cost: " << ct;

  LOG(INFO) << summary.FullReport();
}

TEST(Ceres, SO2Plus) {
  Sophus::SO2d rot1(M_PI / 2);
  Sophus::SO2d rot2(M_PI / 3);
  Sophus::SO2d rot = rot1 * rot2;

  LOG(INFO) << "rot log: " << rot.log();
  LOG(INFO) << "plus log: " << rot1.log() + rot2.log();
  EXPECT_LT(rot.log() - (rot1.log() + rot2.log()), 1e-5);
}

TEST(Ceres, SE2Graph) {
  ceres::Problem problem;
  constexpr int knum_v = 100;
  constexpr double sigma = 0.5;
  // const double sigma = 0.5;
  std::mt19937 rng(2026);
  std::normal_distribution<double> gauss;

  std::vector<Sophus::SE2d, Eigen::aligned_allocator<double>> vertices(knum_v);
  std::vector<Sophus::SE2d, Eigen::aligned_allocator<double>>
      vertices_estimated(knum_v);
  // vertices.reserve(knum_v);
  // vertices_estimated.reserve(knum_v);

  auto random_delta = [&rng, &gauss]() {
    Sophus::SE2d::Tangent twist;
    for (int i = 0; i < twist.size(); ++i) {
      twist[i] = gauss(rng) * sigma;
    }
    return Sophus::SE2d::exp(twist);
  };

  auto *manifold = new Sophus::Manifold<Sophus::SE2>;
  for (int i = 0; i < knum_v; ++i) {
    vertices[i] = Sophus::SE2d::sampleUniform(rng);
    vertices_estimated[i] = vertices[i] * random_delta();
    problem.AddParameterBlock(vertices_estimated[i].data(),
                              Sophus::SE2d::num_parameters, manifold);
  }
  for (int i = 0; i < knum_v; ++i) {
    for (int j = i + 1; j < knum_v; ++j) {
      Sophus::SE2d motion = vertices[i].inverse() * vertices[j];
      ceres::CostFunction *cost_func =
          new ceres::AutoDiffCostFunction<SE2EdgeCost, Sophus::SE2d::DoF,
                                          Sophus::SE2d::num_parameters,
                                          Sophus::SE2d::num_parameters>(
              new SE2EdgeCost(motion, Eigen::Matrix3d::Identity()));
      problem.AddResidualBlock(cost_func, nullptr, vertices_estimated[i].data(),
                               vertices_estimated[j].data());
    }
  }
  ceres::Solver::Options opts;
  opts.max_linear_solver_iterations = 100;
  opts.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);
  LOG(INFO) << summary.FullReport();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  return RUN_ALL_TESTS();
}