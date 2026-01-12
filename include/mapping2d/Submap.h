#pragma once

#include "Frame.h"
#include "LikelihoodField.h"
#include "OGM.h"

#include <memory>
#include <vector>

class Submap {
public:
  Submap(const int img_size, const double resolution, const double patch_radius,
         const double likelihood_sigma, const Sophus::SE2d &submap_pose,
         const size_t id)
      : likelihood_field_(patch_radius, img_size, resolution, likelihood_sigma),
        grid_map_(img_size, resolution), Twl_(submap_pose), id_(id) {}

  bool initialize(std::shared_ptr<Frame> frame);
  bool add_keyframe(std::shared_ptr<Frame> frame);
  bool scan_match(Frame &frame, const LikelihoodField::SolverType solver_type);

  // copy kf from source to current submap
  bool copy_frames(const Submap &source, const size_t kf_num);

  const std::vector<std::shared_ptr<Frame>> &keyframes() const {
    return keyframes_;
  }

  const OccupancyGridMap &grid_map() const { return grid_map_; }
  OccupancyGridMap &grid_map() { return grid_map_; }
  cv::Mat dist_map() const { return likelihood_field_.get_dist_map(); }

  Sophus::SE2d Twl() const { return Twl_; }
  void set_Twl(const Sophus::SE2d &Twl) { Twl_ = Twl; }

  bool has_outsider() const { return grid_map_.has_outsider(); }
  int id() const { return id_; }
  size_t size() const { return keyframes_.size(); };

  void update_kf_pose_w();

private:
  std::vector<std::shared_ptr<Frame>> keyframes_;
  LikelihoodField likelihood_field_;
  OccupancyGridMap grid_map_;
  Sophus::SE2d Twl_;
  int id_ = 0;
};