#pragma once

#include <memory>
#include <vector>

#include "Frame.h"
#include "Submap.h"

class Mapper2d {
public:
  Mapper2d() = default;
  Mapper2d(const int sub_res, const int sub_sz, const int gb_sz,
           const double min_trans, const double min_rotate, const size_t max_kf,
           const int win_sz)
      : submap_resolution_(sub_res), submap_dim_(sub_sz), global_dim_(gb_sz),
        min_kf_translation_(min_trans), min_kf_rotation_(min_rotate),
        max_kf_num_(max_kf), win_size_(win_sz) {}

  bool add_scan(std::unique_ptr<sensor_msgs::msg::LaserScan> scan,
                const LikelihoodField::SolverType solver_type);

  cv::Mat global_map(const int global_dim) const;
  void visualize();
  void save_map(const std::string &map_path, const int occupied_th,
                const int free_th) {
    map_path_ = map_path;
    occupied_th_ = occupied_th;
    free_th_ = free_th;
  };

private:
  std::vector<Submap> submaps_;
  std::shared_ptr<Frame> last_frame_ = nullptr;
  std::shared_ptr<Frame> last_keyframe_ = nullptr;

  int submap_resolution_ = 20;
  int submap_dim_ = 1000;
  int distmap_path_size = 1;
  double likelihood_std = 0.25;

  int global_dim_ = 1000;

  double min_kf_translation_ = 0.3;
  double min_kf_rotation_ = 15 * M_PI / 180;
  size_t max_kf_num_ = 50;
  Sophus::SE2d estimated_motion;
  int win_size_ = 10;

  std::string map_path_;
  int occupied_th_ = 127;
  int free_th_ = 127;
  bool visualize_kf_ = false;

  bool is_keyframe(const Frame &frame) const;
  bool extend_map();

  Submap &cur_submap() { return submaps_.back(); };
};