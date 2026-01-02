#include "visualizer.h"

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace Visualizer {

void Visualize2dScan(const sensor_msgs::msg::LaserScan scan,
                     const Sophus::SE2d &Twf, cv::Mat &img,
                     cv::Vec3b scan_color, bool show_robot, const int img_size,
                     const double resolution, const Sophus::SE2d &Twl) {
  if (!img.data) {
    img = cv::Mat(img_size, img_size, CV_8UC3, cv::Vec3b(255, 255, 255));
  }
  if (img.type() == CV_8UC1) {
    cv::Mat imgc;
    cv::cvtColor(img, imgc, cv::COLOR_GRAY2RGB);
    img = imgc;
  }
  auto pos2img = [&img_size, &resolution](const Eigen::Vector2d pos) {
    return Eigen::Vector2i(pos.x() * resolution + 0.5 * img_size,
                           pos.y() * resolution + 0.5 * img_size);
  };
  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    const double angle = scan.angle_min + scan.angle_increment * i;
    if (angle < scan.angle_min + 30 * M_PI / 180.0 ||
        angle > scan.angle_max - 30 * M_PI / 180.0 ||
        scan.ranges[i] < scan.range_min || scan.ranges[i] > scan.range_max) {
      continue;
    }
    const double x = scan.ranges[i] * std::cos(angle);
    const double y = scan.ranges[i] * std::sin(angle);

    const Eigen::Vector2d Pl = Twl.inverse() * Twf * Eigen::Vector2d(x, y);
    const Eigen::Vector2i scan_img = pos2img(Pl);

    if (scan_img.x() < 0 || scan_img.y() < 0 || scan_img.x() >= img_size ||
        scan_img.y() >= img_size) {
      continue;
    }
    img.at<cv::Vec3b>(scan_img.y(), scan_img.x()) = scan_color;
  }
  if (show_robot) {
    const Sophus::SE2d Tlr = Twl.inverse() * Twf;
    const Eigen::Vector2i robot_img = pos2img(Tlr.translation());
    cv::circle(img, cv::Point2d(robot_img.x(), robot_img.y()), 5,
               cv::Vec3b(255, 0, 0), 1);

    const Eigen::Vector2i arrow_img = pos2img(Tlr * Eigen::Vector2d(1, 0));
    cv::line(img, cv::Point2d(robot_img.x(), robot_img.y()),
             cv::Point2d(arrow_img.x(), arrow_img.y()), cv::Vec3b(255, 0, 0),
             1);
  }
}
} // namespace Visualizer