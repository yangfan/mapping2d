#include "LikelihoodField.h"
#include "OGM.h"
#include "bag_io.hpp"
#include "visualizer.h"

#include <Eigen/Core>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/laser_scan.h>
#include <sophus/se2.hpp>

#include <chrono>

DEFINE_string(map_path, "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/build/",
              "path of map file");
DEFINE_string(bag_path,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "Bag file path");
DEFINE_string(optimizer_type, "G2o", "G2o, Ceres, GN");

TEST(GridMap, LineDrawing) {
  OccupancyGridMap grid_map(1000, 20);
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(700, 600));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(600, 700));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(400, 700));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(300, 600));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(300, 400));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(400, 300));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(600, 300));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(700, 400));

  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(700, 500));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(500, 700));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(300, 500));
  grid_map.draw_line(Eigen::Vector2i(500, 500), Eigen::Vector2i(500, 300));

  cv::Mat map = grid_map.binary_map(127, 127);
  cv::imwrite(FLAGS_map_path + "draw_line.png", map);
}

TEST(GridMap, LocalMapping) {

  auto optimizer_type = LikelihoodField::SolverType::G2o;
  if (FLAGS_optimizer_type == "Ceres") {
    optimizer_type = LikelihoodField::SolverType::Ceres;
  } else if (FLAGS_optimizer_type == "GN") {
    optimizer_type = LikelihoodField::SolverType::GN;
  }

  OccupancyGridMap grid_map(1000, 20);
  BagIO bag_io(FLAGS_bag_path);
  LikelihoodField lf(1, 1000, 20, 0.25);
  std::shared_ptr<Frame> last_frame = nullptr;
  std::shared_ptr<Frame> last_keyframe = nullptr;
  Sophus::SE2d guess{};
  Sophus::SE2d sub_pose{};
  bool submap_finished = false;

  double elapsed = 0.0;
  size_t cnt = 0;

  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
            if (submap_finished) {
              return true;
            }
            auto frame = std::make_shared<Frame>(std::move(scan));
            if (!last_frame) {
              grid_map.add_frame(frame->scan(), Sophus::SE2d());
              lf.set_dist_map(grid_map);
              last_frame = frame;
              last_keyframe = frame;
              return true;
            }

            frame->set_Twf(last_frame->Twf() * guess);
            frame->set_Tlf(last_frame->Tlf());
            Sophus::SE2d aligned_pose = frame->Tlf();
            auto start = std::chrono::steady_clock::now();
            lf.scan2map(frame->scan(), aligned_pose, 10, false, optimizer_type);
            auto end = std::chrono::steady_clock::now();
            elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count();
            cnt++;
            frame->set_Tlf(aligned_pose);
            frame->set_Twf(sub_pose * frame->Tlf());

            Sophus::SE2d rel = last_keyframe->Twf().inverse() * frame->Twf();
            if (rel.translation().norm() > 0.3 ||
                rel.so2().log() > 15 * M_PI / 180) {
              grid_map.add_frame(frame->scan(), frame->Tlf());
              lf.set_dist_map(grid_map);
              last_keyframe = frame;
              LOG(INFO) << "Add frame.";
            }
            cv::Mat img = grid_map.binary_map(122, 132);
            Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), img,
                                        cv::Vec3b(0, 0, 255), true, 1000);
            cv::imshow("grid map", img);

            cv::Mat lf_img = lf.get_dist_map();
            Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), lf_img,
                                        cv::Vec3b(0, 0, 255), true, 1000);
            cv::imshow("likelihood field", lf_img);
            cv::waitKey(20);

            guess = last_frame->Twf().inverse() * frame->Twf();
            last_frame = frame;

            if (grid_map.has_outsider()) {
              cv::imwrite(FLAGS_map_path + FLAGS_optimizer_type + "_submap.png",
                          grid_map.binary_map(122, 132));
              submap_finished = true;
            }

            return true;
          })
      .Process();
  // G2o: average 1.96612 ms
  // Ceres: average 4.33898 ms
  // GN: average 0.0737787 ms
  LOG(INFO) << "scan2map" + FLAGS_optimizer_type << " took average "
            << elapsed / cnt << " ms.";
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}