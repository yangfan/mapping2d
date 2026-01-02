#pragma once

#include <sensor_msgs/msg/laser_scan.hpp>

#include <sophus/se2.hpp>

class Frame {
public:
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

private:
  std::unique_ptr<sensor_msgs::msg::LaserScan> scan_ = nullptr;
  Sophus::SE2d Twf_;
  Sophus::SE2d Tlf_;
};