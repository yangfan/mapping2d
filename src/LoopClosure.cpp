#include "LoopClosure.h"
#include "ceres_types.h"
#include "g2o_types.h"
#include "visualizer.h"

#include <ceres/ceres.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

void LoopClosure::add_submap(Submap *submap) {
  submaps_.emplace_back(submap);
  likelihood_mr_.emplace_back(1000, 20);
  likelihood_mr_.back().set_dist_map(submap->grid_map());
  return;
}

bool LoopClosure::process(Frame *frame) {

  cur_frame_ = frame;
  auto candidates = detect_loop();
  if (candidates.empty()) {
    return false;
  }

  if (!align_submaps(candidates, solver_type_)) {
    return false;
  }

  save_frame();

  optimize();

  return true;
}

std::vector<int> LoopClosure::detect_loop() const {
  std::vector<int> candidates;
  for (const auto submap : submaps_) {
    if (cur_submap_id() - submap->id() <= min_gap_ ||
        loop_constraints_.find(std::make_pair(submap->id(), cur_submap_id())) !=
            loop_constraints_.end()) {
      continue;
    }
    const Sophus::SE2d Tlf = submap->Twl().inverse() * cur_frame_->Twf();
    if (Tlf.translation().norm() < max_displacement_) {
      candidates.emplace_back(submap->id());
    }
  }
  return candidates;
}

bool LoopClosure::align_submaps(const std::vector<int> candidates,
                                const LikelihoodField::SolverType type) {
  bool submap_aligned = false;
  for (const int submap_id : candidates) {
    auto &lf_mr = likelihood_mr_[submap_id];
    lf_mr.set_dist_map(submaps_[submap_id]->grid_map());

    Sophus::SE2d T_l1_f =
        submaps_[submap_id]->Twl().inverse() * cur_frame_->Twf();
    if (lf_mr.scan2map(cur_frame_->scan(), T_l1_f, 10, false, type)) {

      const Sophus::SE2d T_l1_l2 =
          T_l1_f * cur_frame_->Twf().inverse() * submaps_.back()->Twl();
      loop_constraints_.insert(
          {std::make_pair(submap_id, cur_submap_id()),
           LoopConstraint(T_l1_l2, cur_frame_->kf_id(), T_l1_f)});
      submap_aligned = true;

      LOG(INFO) << "Loop Detected, " << "Submap " << submap_id << " to Submap "
                << cur_submap_id() << ", to Frame " << cur_frame_->kf_id()
                << ": translation: " << T_l1_f.translation().transpose()
                << ", rotation: " << T_l1_f.so2().log();
      if (ofs_.is_open()) {
        ofs_ << submap_id << " " << cur_submap_id() << " "
             << T_l1_l2.translation().x() << " " << T_l1_l2.translation().y()
             << " " << T_l1_l2.so2().log() << " " << cur_frame_->kf_id() << " "
             << T_l1_f.translation().x() << " " << T_l1_f.translation().y()
             << " " << T_l1_f.so2().log() << std::endl;
      }
      visualize(T_l1_f, submap_id);
    }
  }
  return submap_aligned;
}

bool LoopClosure::optimize() {
  switch (solver_type_) {
  case LikelihoodField::SolverType::G2o:
    return optimize_G2o();
  case LikelihoodField::SolverType::Ceres:
    return optimize_Ceres();
  default:
    return optimize_G2o();
  }
  return false;
}

bool LoopClosure::optimize_G2o() {
  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>>;
  using LinearSolverType =
      g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

  auto *solver = new g2o::OptimizationAlgorithmLevenberg(
      std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(solver);

  for (size_t sid = 0; sid < submaps_.size(); ++sid) {
    const auto &submap = submaps_[sid];
    auto *vertex = new VertexSE2();
    vertex->setId(sid);
    vertex->setEstimate(submap->Twl());
    optimizer.addVertex(vertex);

    if (vertex->id() == 0) {
      vertex->setFixed(true);
    } else {
      auto *consecutive_edge = new EdgeSubmaps();
      consecutive_edge->setId(optimizer.edges().size());
      consecutive_edge->setVertex(0, optimizer.vertex(sid - 1));
      consecutive_edge->setVertex(1, vertex);
      consecutive_edge->setMeasurement(submaps_[sid - 1]->Twl().inverse() *
                                       submap->Twl());
      consecutive_edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
      optimizer.addEdge(consecutive_edge);
    }
  }

  std::vector<EdgeSubmaps *> loop_edges;
  loop_edges.reserve(loop_constraints_.size());
  for (const auto &[match_ids, constraint] : loop_constraints_) {

    auto *loop_edge = new EdgeSubmaps();
    loop_edge->setId(optimizer.edges().size());
    loop_edge->setVertex(0, optimizer.vertex(match_ids.first));
    loop_edge->setVertex(1, optimizer.vertex(match_ids.second));
    loop_edge->setMeasurement(constraint.T_l1_l2);
    loop_edge->setInformation(Eigen::Matrix3d::Identity());

    auto *rk = new g2o::RobustKernelCauchy();
    rk->setDelta(rk_delta_loop_constraint);
    loop_edge->setRobustKernel(rk);
    optimizer.addEdge(loop_edge);

    loop_edges.emplace_back(loop_edge);
  }

  optimizer.setVerbose(verbose_);
  optimizer.initializeOptimization();
  optimizer.optimize(iterations_);

  for (auto *edge : loop_edges) {
    if (edge->chi2() > rk_delta_loop_constraint) {
      edge->setLevel(1);
      loop_constraints_.erase(
          std::pair<int, int>(edge->vertex(0)->id(), edge->vertex(1)->id()));

      LOG(INFO) << "sub" << edge->vertex(0)->id() << " to sub"
                << edge->vertex(1)->id() << " invalid ch2: " << edge->chi2();

    } else {
      edge->setRobustKernel(nullptr);

      LOG(INFO) << "sub" << edge->vertex(0)->id() << " to sub"
                << edge->vertex(1)->id() << " valid ch2: " << edge->chi2();
    }
  }

  optimizer.optimize(iterations_);

  for (const auto &submap : submaps_) {
    submap->set_Twl(
        static_cast<VertexSE2 *>(optimizer.vertex(submap->id()))->estimate());
    submap->update_kf_pose_w();
  }

  return true;
}

bool LoopClosure::optimize_Ceres() {

  ceres::Problem problem;

  std::vector<Sophus::SE2d, Eigen::aligned_allocator<Sophus::SE2d>>
      submap_poses(submaps_.size());

  // consecutive edges
  auto *manifold = new Sophus::Manifold<Sophus::SE2>;
  for (size_t sid = 0; sid < submaps_.size(); ++sid) {

    const auto &submap = submaps_[sid];
    submap_poses[sid] = submap->Twl();
    problem.AddParameterBlock(submap_poses[sid].data(),
                              Sophus::SE2d::num_parameters, manifold);

    if (sid == 0) {
      problem.SetParameterBlockConstant(submap_poses[0].data());
    } else {
      ceres::CostFunction *cost_func =
          new ceres::AutoDiffCostFunction<SE2EdgeCost, Sophus::SE2d::DoF,
                                          Sophus::SE2d::num_parameters,
                                          Sophus::SE2d::num_parameters>(
              new SE2EdgeCost(submaps_[sid - 1]->Twl().inverse() *
                                  submaps_[sid]->Twl(),
                              1e2 * Eigen::Matrix3d::Identity()));

      problem.AddResidualBlock(cost_func, nullptr, submap_poses[sid - 1].data(),
                               submap_poses[sid].data());
    }
  }

  std::vector<ceres::ResidualBlockId> loop_edges;
  loop_edges.reserve(loop_constraints_.size());
  std::vector<ceres::LossFunctionWrapper *> loop_loss;
  loop_loss.reserve(loop_constraints_.size());
  std::vector<std::pair<int, int>> loop_matches;
  loop_matches.reserve(loop_constraints_.size());

  // loop edges
  for (const auto &[match_ids, constraint] : loop_constraints_) {
    ceres::CostFunction *cost_func =
        new ceres::AutoDiffCostFunction<SE2EdgeCost, Sophus::SE2d::DoF,
                                        Sophus::SE2d::num_parameters,
                                        Sophus::SE2d::num_parameters>(
            new SE2EdgeCost(constraint.T_l1_l2, Eigen::Matrix3d::Identity()));

    ceres::LossFunctionWrapper *loss_func = new ceres::LossFunctionWrapper(
        new ceres::CauchyLoss(rk_delta_loop_constraint), ceres::TAKE_OWNERSHIP);

    auto loop_residual = problem.AddResidualBlock(
        cost_func, loss_func, submap_poses[match_ids.first].data(),
        submap_poses[match_ids.second].data());

    loop_edges.emplace_back(loop_residual);
    loop_loss.emplace_back(loss_func);
    loop_matches.emplace_back(match_ids);
  }
  ceres::Solver::Options opts;
  opts.minimizer_progress_to_stdout = false;
  opts.max_linear_solver_iterations = iterations_;
  opts.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
  opts.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  for (size_t lid = 0; lid < loop_edges.size(); ++lid) {
    ceres::ResidualBlockId loop_edge = loop_edges[lid];
    double cost = 0;
    problem.EvaluateResidualBlockAssumingParametersUnchanged(
        loop_edge, false, &cost, nullptr, nullptr);

    // verify outlier: ch2 or 2 * cost > rk_delta
    if (2 * cost > rk_delta_loop_constraint) {
      // remove outlier residual block
      problem.RemoveResidualBlock(loop_edge);
      loop_constraints_.erase(loop_matches[lid]);

      LOG(INFO) << "sub" << loop_matches[lid].first << " to sub"
                << loop_matches[lid].second << " invalid chi2: " << 2 * cost;
    } else {
      // disable rk
      auto *loss_func = loop_loss[lid];
      loss_func->Reset(new ceres::TrivialLoss, ceres::TAKE_OWNERSHIP);
      LOG(INFO) << "sub" << loop_matches[lid].first << " to sub"
                << loop_matches[lid].second << " valid chi2: " << 2 * cost;
    }
  }
  // Refine
  ceres::Solve(opts, &problem, &summary);

  // asign back the optimized submap poses
  for (size_t sid = 0; sid < submaps_.size(); ++sid) {
    submaps_[sid]->set_Twl(submap_poses[sid]);
    submaps_[sid]->update_kf_pose_w();
  }

  return true;
}

void LoopClosure::visualize(const Sophus::SE2d scan_pose,
                            const int submap_id) const {
  cv::Mat img = submaps_[submap_id]->grid_map().binary_map(127, 127);
  Visualizer::Visualize2dScan(cur_frame_->scan(), scan_pose, img,
                              cv::Vec3b(0, 0, 255), true, 1000, 20);
  cv::imshow("scan to submap", img);

  cv::Mat aligned_subs = Visualizer::global_map(
      std::vector<Submap *>{submaps_.back(), submaps_[submap_id]});
  cv::imshow("aligned submaps", aligned_subs);

  cv::waitKey(10);
}

bool LoopClosure::save_frame() const {
  return cur_frame_->save(
      "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/build/maps/f" +
      std::to_string(cur_frame_->kf_id()) + ".txt");
}