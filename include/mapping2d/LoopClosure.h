#pragma once

#include "Frame.h"
#include "LikelihoodFieldMR.h"
#include "Submap.h"

#include <sophus/se2.hpp>

#include <map>
#include <vector>

class LoopClosure {
public:
  struct LoopConstraint {
    LoopConstraint(const Sophus::SE2d &submap_pose, const size_t kf_id,
                   const Sophus::SE2d &frame_pose)
        : T_l1_l2(submap_pose), kf_id(kf_id), T_l1_f(frame_pose) {}
    Sophus::SE2d T_l1_l2;
    size_t kf_id = 0;
    Sophus::SE2d T_l1_f;
  };
  LoopClosure() = default;
  LoopClosure(const int min_gap, const double max_dis,
              const LikelihoodField::SolverType type)
      : min_gap_(min_gap), max_displacement_(max_dis), solver_type_(type) {}

  void add_submap(Submap *submap);
  bool process(Frame *frame);

  void visualize(const Sophus::SE2d scan_pose, const int submap_id) const;

  const std::map<std::pair<int, int>, LoopConstraint> &
  loop_constraints() const {
    return loop_constraints_;
  }

  const LikelihoodFieldMR &likelihood_mr(size_t id) const {
    return likelihood_mr_[id];
  }

  bool record_loops(const std::string &info_path) {
    ofs_.open(info_path);
    return ofs_.is_open();
  }

private:
  std::vector<Submap *> submaps_;
  std::vector<LikelihoodFieldMR> likelihood_mr_;
  std::map<std::pair<int, int>, LoopConstraint> loop_constraints_;
  Frame *cur_frame_ = nullptr;
  std::ofstream ofs_;

  int min_gap_ = 1;
  double max_displacement_ = 15.0;

  double rk_delta_loop_constraint = 1.0;
  bool verbose_ = false;
  int iterations_ = 10;

  LikelihoodField::SolverType solver_type_ = LikelihoodField::SolverType::G2o;

  std::vector<int> detect_loop() const;
  bool align_submaps(const std::vector<int> candidates,
                     const LikelihoodField::SolverType type);
  bool optimize();
  bool optimize_G2o();
  bool optimize_Ceres();

  int cur_submap_id() const { return submaps_.back()->id(); }

  bool save_frame() const;
};