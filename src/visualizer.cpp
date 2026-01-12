#include "visualizer.h"

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <execution>

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

cv::Mat global_map(const std::vector<Submap *> &submaps, const int map_dim) {
  constexpr int submap_dim = 1000;
  constexpr int submap_resolution_ = 20;
  constexpr int free_th = 127;
  constexpr int occupied_th = 127;

  Eigen::Vector2d top_left(std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max());
  Eigen::Vector2d bottom_right(std::numeric_limits<double>::lowest(),
                               std::numeric_limits<double>::lowest());
  const double sub_sz = double(submap_dim) / double(submap_resolution_);

  for (const auto &sub : submaps) {
    const Eigen::Vector2d pos = sub->Twl().translation();
    const Eigen::Vector2d sub_tl =
        pos - Eigen::Vector2d(0.5 * sub_sz, 0.5 * sub_sz);
    const Eigen::Vector2d sub_br =
        pos + Eigen::Vector2d(0.5 * sub_sz, 0.5 * sub_sz);
    top_left.x() = std::min(top_left.x(), sub_tl.x());
    top_left.y() = std::min(top_left.y(), sub_tl.y());
    bottom_right.x() = std::max(bottom_right.x(), sub_br.x());
    bottom_right.y() = std::max(bottom_right.y(), sub_br.y());
  }

  const double global_width = bottom_right.x() - top_left.x();
  const double global_height = bottom_right.y() - top_left.y();
  if (global_height <= 0 || global_width <= 0) {
    return cv::Mat();
  }

  const Eigen::Vector2d global_center =
      top_left + Eigen::Vector2d(0.5 * global_width, 0.5 * global_height);
  const double global_resolution =
      map_dim / std::max(global_height, global_width);

  const int global_img_width = std::ceil(global_width * global_resolution);
  const int global_img_height = std::ceil(global_height * global_resolution);
  const Eigen::Vector2d global_img_center(0.5 * global_img_width,
                                          0.5 * global_img_height);
  std::vector<Eigen::Vector2i> idx;
  idx.reserve(global_img_height * global_img_width);
  for (int y = 0; y < global_img_height; ++y) {
    for (int x = 0; x < global_img_width; ++x) {
      idx.emplace_back(x, y);
    }
  }

  cv::Mat global_map(global_img_height, global_img_width, CV_8UC3,
                     cv::Scalar(127, 127, 127));

  auto img2world = [&global_img_center, &global_center, &global_resolution](
                       const Eigen::Vector2i coord) -> Eigen::Vector2d {
    return (coord.cast<double>() - global_img_center) / global_resolution +
           global_center;
  };
  std::for_each(
      std::execution::par_unseq, idx.begin(), idx.end(),
      [&](const Eigen::Vector2i &coord) {
        const Eigen::Vector2d pw = img2world(coord);

        for (const auto &sub : submaps) {
          const Eigen::Vector2d pl = sub->Twl().inverse() * pw; // in submap
          const Eigen::Vector2i sub_coord =
              (pl * submap_resolution_ +
               0.5 * Eigen::Vector2d(submap_dim, submap_dim))
                  .cast<int>();

          if (sub_coord.x() < 0 || sub_coord.x() >= submap_dim ||
              sub_coord.y() < 0 || sub_coord.y() >= submap_dim) {
            continue;
          }

          const uchar value =
              sub->grid_map().map().at<uchar>(sub_coord.y(), sub_coord.x());
          if (value > free_th) {
            global_map.at<cv::Vec3b>(coord.y(), coord.x()) =
                cv::Vec3b(255, 255, 255);
            break;
          } else if (value < occupied_th) {
            global_map.at<cv::Vec3b>(coord.y(), coord.x()) = cv::Vec3b(0, 0, 0);
            break;
          }
        }
      });

  auto world2img = [&global_img_center, &global_center, &global_resolution](
                       const Eigen::Vector2d pos_w) -> Eigen::Vector2d {
    return (pos_w - global_center) * global_resolution + global_img_center;
  };
  for (const auto &sub : submaps) {
    const Eigen::Vector2d xaxis = world2img(sub->Twl() * Eigen::Vector2d(1, 0));
    const Eigen::Vector2d yaxis = world2img(sub->Twl() * Eigen::Vector2d(0, 1));
    const Eigen::Vector2d origin = world2img(sub->Twl().translation());
    cv::line(global_map, cv::Point2d(origin.x(), origin.y()),
             cv::Point2d(xaxis.x(), xaxis.y()), cv::Vec3b(0, 0, 255), 2);
    cv::line(global_map, cv::Point2d(origin.x(), origin.y()),
             cv::Point2d(yaxis.x(), yaxis.y()), cv::Vec3b(255, 0, 0), 2);
    cv::putText(global_map, std::to_string(sub->id()),
                cv::Point2d(origin.x() - 10, origin.y() - 10),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Vec3b(0, 255, 0));

    for (const auto &kf : sub->keyframes()) {
      const Eigen::Vector2d kf_coord = world2img(kf->Twf().translation());
      cv::circle(global_map, cv::Point2d(kf_coord.x(), kf_coord.y()), 1,
                 cv::Vec3d(0, 0, 255));
    }
  }

  return global_map;
}
} // namespace Visualizer