#include "Mapper2d.h"
#include "visualizer.h"

#include <algorithm>
#include <execution>

bool Mapper2d::add_scan(std::unique_ptr<sensor_msgs::msg::LaserScan> scan,
                        const LikelihoodField::SolverType solver_type) {
  if (scan->ranges.empty()) {
    return false;
  }
  auto frame = std::make_shared<Frame>(std::move(scan));
  if (!last_frame_) {
    submaps_.emplace_back(submap_dim_, submap_resolution_, distmap_path_size,
                          likelihood_std, Sophus::SE2d(), 0);
    cur_submap().initialize(frame);
    last_frame_ = frame;
    last_keyframe_ = frame;
    visualize_kf_ = true;
    return true;
  }
  frame->set_Tlf(last_frame_->Tlf());
  frame->set_Twf(last_frame_->Twf() * estimated_motion);
  if (!cur_submap().scan_match(*frame, solver_type)) {
    return false;
  }
  if (is_keyframe(*frame)) {
    cur_submap().add_keyframe(frame);
    last_keyframe_ = frame;
    visualize_kf_ = true;
  }
  estimated_motion = last_frame_->Twf().inverse() * frame->Twf();
  last_frame_ = frame;

  if (cur_submap().has_outsider() || cur_submap().size() > max_kf_num_) {
    return extend_map();
  }

  return true;
}

bool Mapper2d::is_keyframe(const Frame &frame) const {

  Sophus::SE2d rel_motion = last_keyframe_->Twf().inverse() * frame.Twf();
  return submaps_.empty() ||
         rel_motion.translation().norm() > min_kf_translation_ ||
         std::abs(rel_motion.so2().log()) > min_kf_rotation_;
}

bool Mapper2d::extend_map() {
  if (submaps_.empty()) {
    return false;
  }
  if (!map_path_.empty()) {
    cv::Mat grid_map =
        cur_submap().grid_map().binary_map(occupied_th_, free_th_);
    cv::imwrite(map_path_ + "submap" + std::to_string(cur_submap().id()) +
                    ".png",
                grid_map);
  }
  submaps_.emplace_back(submap_dim_, submap_resolution_, distmap_path_size,
                        likelihood_std, last_frame_->Twf(), submaps_.size());
  cur_submap().copy_frames(*(submaps_.rbegin() + 1), win_size_);
  return true;
}

cv::Mat Mapper2d::global_map(const int map_dim) const {

  Eigen::Vector2d top_left(std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::max());
  Eigen::Vector2d bottom_right(std::numeric_limits<double>::lowest(),
                               std::numeric_limits<double>::lowest());
  const double sub_sz = double(submap_dim_) / double(submap_resolution_);

  for (const auto &sub : submaps_) {
    const Eigen::Vector2d pos = sub.Twl().translation();
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

        for (const auto &sub : submaps_) {
          const Eigen::Vector2d pl = sub.Twl().inverse() * pw; // in submap
          const Eigen::Vector2i sub_coord =
              (pl * submap_resolution_ +
               0.5 * Eigen::Vector2d(submap_dim_, submap_dim_))
                  .cast<int>();

          if (sub_coord.x() < 0 || sub_coord.x() >= submap_dim_ ||
              sub_coord.y() < 0 || sub_coord.y() >= submap_dim_) {
            continue;
          }

          const uchar value =
              sub.grid_map().map().at<uchar>(sub_coord.y(), sub_coord.x());
          if (value > free_th_) {
            global_map.at<cv::Vec3b>(coord.y(), coord.x()) =
                cv::Vec3b(255, 255, 255);
            break;
          } else if (value < occupied_th_) {
            global_map.at<cv::Vec3b>(coord.y(), coord.x()) = cv::Vec3b(0, 0, 0);
            break;
          }
        }
      });

  auto world2img = [&global_img_center, &global_center, &global_resolution](
                       const Eigen::Vector2d pos_w) -> Eigen::Vector2d {
    return (pos_w - global_center) * global_resolution + global_img_center;
  };
  for (const auto &sub : submaps_) {
    const Eigen::Vector2d xaxis = world2img(sub.Twl() * Eigen::Vector2d(1, 0));
    const Eigen::Vector2d yaxis = world2img(sub.Twl() * Eigen::Vector2d(0, 1));
    const Eigen::Vector2d origin = world2img(sub.Twl().translation());
    cv::line(global_map, cv::Point2d(origin.x(), origin.y()),
             cv::Point2d(xaxis.x(), xaxis.y()), cv::Vec3b(0, 0, 255), 2);
    cv::line(global_map, cv::Point2d(origin.x(), origin.y()),
             cv::Point2d(yaxis.x(), yaxis.y()), cv::Vec3b(255, 0, 0), 2);
    cv::putText(global_map, std::to_string(sub.id()),
                cv::Point2d(origin.x() - 10, origin.y() - 10),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Vec3b(0, 255, 0));

    for (const auto &kf : sub.keyframes()) {
      const Eigen::Vector2d kf_coord = world2img(kf->Twf().translation());
      cv::circle(global_map, cv::Point2d(kf_coord.x(), kf_coord.y()), 1,
                 cv::Vec3d(0, 0, 255));
    }
  }

  return global_map;
}

void Mapper2d::visualize() {
  cv::Mat grid_map = cur_submap().grid_map().binary_map(126, 130);
  Visualizer::Visualize2dScan(last_frame_->scan(), last_frame_->Twf(), grid_map,
                              cv::Vec3b(0, 0, 255), true, 1000, 20,
                              cur_submap().Twl());
  cv::imshow("submap", grid_map);

  cv::Mat dist_map = cur_submap().dist_map();
  Visualizer::Visualize2dScan(last_frame_->scan(), last_frame_->Twf(), dist_map,
                              cv::Vec3b(0, 0, 255), true, 1000, 20,
                              cur_submap().Twl());
  cv::imshow("distance map", dist_map);

  if (visualize_kf_) {
    cv::imshow("global map", global_map(global_dim_));
    visualize_kf_ = false;
  }

  cv::waitKey(20);
}
