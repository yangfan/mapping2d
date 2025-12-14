#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <random>
#include <string>

#include "bag_io.hpp"
#include "icp2d.h"
#include "visualizer.h"

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");
DEFINE_string(visual_optimizer, "GN", "G2o, Ceres, GN");

TEST(ICP2DP2LTest, LineFitting) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> normal(0, 0.01);

  Eigen::Vector3d line_coeffs(1, 2, 3);
  std::vector<ICP2D::Point> points;
  points.reserve(100);
  for (size_t i = 0; i < 100; ++i) {
    const double x = uniform(rng);
    const double y =
        normal(rng) - (line_coeffs[0] * x + line_coeffs[2]) / line_coeffs[1];
    points.emplace_back(x, y);
  }
  Eigen::Vector3d estimate_line;
  EXPECT_TRUE(ICP2D::line_fitting(points, estimate_line));
  auto err = [&estimate_line](ICP2D::Point pt) {
    return estimate_line[0] * pt.x() + estimate_line[1] * pt.y() +
           estimate_line[2];
  };
  for (const auto &pt : points) {
    EXPECT_LE(err(pt), 0.05);
  }
  LOG(INFO) << "Line coefficients: " << line_coeffs.transpose().normalized()
            << ", line estimate: " << estimate_line.transpose().normalized();
}

TEST(ICP2DP2LTest, BasicP2LGN) {
  BagIO bag_io(FLAGS_bag_file);
  size_t cnt = 0;
  sensor_msgs::msg::LaserScan target;
  sensor_msgs::msg::LaserScan source;
  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&cnt, &target, &source](sensor_msgs::msg::LaserScan scan) {
            if (cnt == 0) {
              target = scan;
            } else if (cnt == 1) {
              source = scan;
            }
            cnt++;
            return true;
          })
      .Process();
  Sophus::SE2d Tts;

  EXPECT_TRUE(ICP2D::P2LGN(target, source, Tts, 10, true));
  LOG(INFO) << "translation: " << Tts.translation().transpose()
            << ", ang: " << Tts.so2().log();
  EXPECT_LE(Tts.translation().norm(), 0.01);
  EXPECT_LE(std::abs(Tts.so2().log()), 0.01);
}

TEST(ICP2DP2LTest, VisualP2L) {
  BagIO bag_io(FLAGS_bag_file);
  sensor_msgs::msg::LaserScan last_scan;
  bool initialized = false;

  ICP2D::OptimizerType opt_type = ICP2D::OptimizerType::GN;
  if (FLAGS_visual_optimizer == "G2o") {
    opt_type = ICP2D::OptimizerType::G2o;
  } else if (FLAGS_visual_optimizer == "Ceres") {
    opt_type = ICP2D::OptimizerType::Ceres;
  }

  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&initialized, &last_scan,
           &opt_type](sensor_msgs::msg::LaserScan scan) {
            if (!initialized) {
              last_scan = scan;
              initialized = true;
              return true;
            }

            Sophus::SE2d pose;
            EXPECT_TRUE(ICP2D::P2L(last_scan, scan, pose, 10, false, opt_type));
            cv::Mat img;
            Visualizer::Visualize2dScan(last_scan, Sophus::SE2d(), img,
                                        cv::Vec3b(255, 0, 0), false);
            Visualizer::Visualize2dScan(scan, pose, img, cv::Vec3b(0, 0, 255),
                                        true);
            cv::imshow("ICP2d: Point to Point", img);
            cv::waitKey(20);
            last_scan = scan;
            return true;
          })
      .Process();
  SUCCEED();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}