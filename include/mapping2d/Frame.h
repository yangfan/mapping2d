#pragma once

#include <sensor_msgs/msg/laser_scan.hpp>

#include <sophus/se2.hpp>

#include <fstream>
#include <unordered_map>

class Frame {
public:
  Frame() : Frame(nullptr) {}
  explicit Frame(std::unique_ptr<sensor_msgs::msg::LaserScan> scan)
      : scan_(std::move(scan)) {}
  Frame(std::unique_ptr<sensor_msgs::msg::LaserScan> scan,
        const Sophus::SE2d &Twf, const Sophus::SE2d &Tlf)
      : scan_(std::move(scan)), Twf_(Twf), Tlf_(Tlf) {}

  const sensor_msgs::msg::LaserScan &scan() const { return *scan_; }
  Sophus::SE2d Twf() const { return Twf_; }
  Sophus::SE2d Tlf() const { return Tlf_; }
  void set_Twf(const Sophus::SE2d &Twf) { Twf_ = Twf; }
  void set_Tlf(const Sophus::SE2d &Tlf) { Tlf_ = Tlf; }

  void set_kf_id(const size_t id) { kf_id_ = id; }
  size_t kf_id() const { return kf_id_; }

  bool save(const std::string file) const {
    std::ofstream ofs(file);
    if (!ofs.is_open()) {
      return false;
    }
    ofs << kf_id_ << std::endl;
    ofs << Twf_.translation().x() << " " << Twf_.translation().y() << " "
        << Twf_.so2().log() << std::endl;
    ofs << scan_->angle_min << " " << scan_->angle_max << " "
        << scan_->angle_increment << std::endl;
    ofs << scan_->range_min << " " << scan_->range_max << " "
        << scan_->ranges.size() << std::endl;
    for (const auto range : scan_->ranges) {
      ofs << range << " ";
    }
    ofs.close();
    return true;
  }

  bool load(const std::string &file) {
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
      return false;
    }
    ifs >> kf_id_;
    double x = 0, y = 0, theta = 0;
    ifs >> x >> y >> theta;
    Twf_ = Sophus::SE2d(theta, Eigen::Vector2d(x, y));
    scan_ = std::make_unique<sensor_msgs::msg::LaserScan>();
    ifs >> scan_->angle_min >> scan_->angle_max >> scan_->angle_increment;
    size_t range_sz = 0;
    ifs >> scan_->range_min >> scan_->range_max >> range_sz;
    scan_->ranges.reserve(range_sz);
    for (size_t i = 0; i < range_sz; ++i) {
      double r = 0;
      ifs >> r;
      scan_->ranges.emplace_back(r);
    }
    return true;
  }

  void set_local_pose(const int submap_id, const Sophus::SE2d Tlf) {
    Tlfs_[submap_id] = Tlf;
  }
  const Sophus::SE2d &local_pose(const int submap_id) const {
    return Tlfs_.at(submap_id);
  }

private:
  std::unique_ptr<sensor_msgs::msg::LaserScan> scan_ = nullptr;
  Sophus::SE2d Twf_;
  // local pose on the last asigned submap
  Sophus::SE2d Tlf_;
  // local pose on the specified submap
  std::unordered_map<int, Sophus::SE2d> Tlfs_;
  size_t kf_id_ = 0;
};