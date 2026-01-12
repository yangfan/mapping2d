#include "Submap.h"

bool Submap::initialize(std::shared_ptr<Frame> frame) {
  keyframes_.clear();
  grid_map_.reset();

  frame->set_Tlf(Sophus::SE2d());
  frame->set_Twf(Sophus::SE2d());
  frame->set_local_pose(id_, frame->Tlf());

  grid_map_.add_frame(*frame);
  likelihood_field_.set_dist_map(grid_map_);

  keyframes_.emplace_back(frame);
  return true;
}

bool Submap::copy_frames(const Submap &source, const size_t kf_num) {
  if (source.keyframes_.size() < kf_num) {
    return false;
  }

  for (size_t fid = source.keyframes_.size() - kf_num;
       fid < source.keyframes_.size(); ++fid) {
    auto keyframe = source.keyframes_[fid];
    keyframe->set_Tlf(Twl_.inverse() * keyframe->Twf());
    keyframe->set_local_pose(id_, keyframe->Tlf());
    grid_map_.add_frame(*keyframe);
    keyframes_.emplace_back(keyframe);
  }
  likelihood_field_.set_dist_map(grid_map_);

  return true;
}

bool Submap::add_keyframe(std::shared_ptr<Frame> frame) {

  grid_map_.add_frame(*frame);
  likelihood_field_.set_dist_map(grid_map_);
  keyframes_.emplace_back(frame);
  frame->set_local_pose(id_, frame->Tlf());
  return true;
}

bool Submap::scan_match(Frame &frame,
                        const LikelihoodField::SolverType solver_type) {
  if (keyframes_.empty()) {
    return false;
  }
  Sophus::SE2d Tts = frame.Tlf();

  if (!likelihood_field_.scan2map(frame.scan(), Tts, 10, false, solver_type)) {
    return false;
  }
  frame.set_Tlf(Tts);
  frame.set_Twf(Twl_ * frame.Tlf());

  return true;
}

void Submap::update_kf_pose_w() {
  for (const auto &frame : keyframes_) {
    frame->set_Twf(Twl_ * frame->local_pose(id_));
  }
}