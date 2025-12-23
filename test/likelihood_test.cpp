#include "LikelihoodField.h"
#include "bag_io.hpp"
#include "visualizer.h"

#include <chrono>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");
DEFINE_string(
    dist_map_file,
    "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/data/output/dist_map.png",
    "path of distance map file");
DEFINE_string(likelihood_field_file,
              "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/data/output/"
              "likelihood_field.png",
              "path of distance map file");
DEFINE_string(optimizer_type, "GN", "G2o, Ceres, GN");

TEST(Likelihood, DistMap) {
  BagIO bag_io(FLAGS_bag_file);
  LikelihoodField likelihood_field(1, 1000, 20, 0.25);
  bool scan_obtained = false;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&scan_obtained, &likelihood_field](
                           const sensor_msgs::msg::LaserScan &scan) {
                         if (!scan_obtained) {
                           likelihood_field.set_dist_map(scan);
                           scan_obtained = true;
                         }
                         return true;
                       })
      .Process();
  likelihood_field.get_dist_map(FLAGS_dist_map_file);
  SUCCEED();
}

TEST(Likelihood, LikelihoodField) {
  BagIO bag_io(FLAGS_bag_file);
  LikelihoodField likelihood_field(1, 1000, 20, 0.25);
  bool scan_obtained = false;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&scan_obtained, &likelihood_field](
                           const sensor_msgs::msg::LaserScan &scan) {
                         if (!scan_obtained) {
                           likelihood_field.set_dist_map(scan);
                           scan_obtained = true;
                         }
                         return true;
                       })
      .Process();
  likelihood_field.get_likelihood_field(FLAGS_likelihood_field_file);
  SUCCEED();
}

TEST(Likelihood, ALignBasic) {
  LikelihoodField::SolverType solver_type = LikelihoodField::SolverType::GN;
  if (FLAGS_optimizer_type == "G2o") {
    solver_type = LikelihoodField::SolverType::G2o;
  } else if (FLAGS_optimizer_type == "Ceres") {
    solver_type = LikelihoodField::SolverType::Ceres;
  }
  BagIO bag_io(FLAGS_bag_file);
  size_t cnt = 0;
  sensor_msgs::msg::LaserScan target;
  sensor_msgs::msg::LaserScan source;
  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&cnt, &target, &source](const sensor_msgs::msg::LaserScan &scan) {
            if (cnt == 0) {
              target = scan;
            } else if (cnt == 1) {
              source = scan;
            }
            cnt++;
            return true;
          })
      .Process();
  LikelihoodField lf(1, 1000, 20, 0.25);
  lf.set_dist_map(target);
  Sophus::SE2d pose;
  EXPECT_TRUE(lf.align(target, source, pose, 10, true, solver_type));
  LOG(INFO) << "translation: " << pose.translation().transpose()
            << ", angle: " << pose.so2().log();
  SUCCEED();
}

TEST(Likelihood, AlignTest) {
  LikelihoodField::SolverType solver_type = LikelihoodField::SolverType::GN;
  if (FLAGS_optimizer_type == "G2o") {
    solver_type = LikelihoodField::SolverType::G2o;
  } else if (FLAGS_optimizer_type == "Ceres") {
    solver_type = LikelihoodField::SolverType::Ceres;
  }
  BagIO bag_io(FLAGS_bag_file);
  LikelihoodField lf(1, 1000, 20, 0.25);
  sensor_msgs::msg::LaserScan last_scan;
  bool initialized = false;
  double elapsed = 0;
  size_t cnt = 0;
  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&lf, &last_scan, &initialized, &elapsed, &cnt,
           &solver_type](const sensor_msgs::msg::LaserScan &scan) {
            if (!initialized) {
              last_scan = scan;
              initialized = true;
              return true;
            }
            Sophus::SE2d pose;
            auto start = std::chrono::steady_clock::now();
            EXPECT_TRUE(
                lf.align(last_scan, scan, pose, 10, false, solver_type));
            auto end = std::chrono::steady_clock::now();
            elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count();
            cnt++;
            cv::Mat img;
            Visualizer::Visualize2dScan(last_scan, Sophus::SE2d(), img,
                                        cv::Vec3b(255, 0, 0), false, 1000, 20);
            Visualizer::Visualize2dScan(scan, pose, img, cv::Vec3b(0, 0, 255),
                                        true, 1000, 20);
            cv::imshow("Ceres likelihood field alignment: " +
                           FLAGS_optimizer_type,
                       img);

            cv::Mat dist_map = lf.get_dist_map();
            // cv::Mat dist_map = lf.get_likelihood_field();
            Visualizer::Visualize2dScan(last_scan, Sophus::SE2d(), dist_map,
                                        cv::Vec3b(255, 0, 0), false, 1000, 20);
            cv::imshow("distance map", dist_map);

            cv::waitKey(20);
            last_scan = scan;
            return true;
          })
      .Process();
  // GN: average 3.64123 ms
  // G2o: average 6.62385 ms
  // Ceres: average 10.2878 ms
  LOG(INFO) << FLAGS_optimizer_type << " Alignment tooks average "
            << elapsed / cnt << " ms.";
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}