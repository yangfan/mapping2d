#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>

#include "bag_io.hpp"
#include "icp2d.h"
#include "visualizer.h"

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");
DEFINE_string(visual_optimizer, "G2o", "G2o, Ceres, GN");

TEST(ICP2DTest, BasicP2PGN) {
  BagIO bag_io(FLAGS_bag_file);
  size_t cnt = 0;
  std::unique_ptr<sensor_msgs::msg::LaserScan> target;
  std::unique_ptr<sensor_msgs::msg::LaserScan> source;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&cnt, &target, &source](
                           std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         if (cnt == 0) {
                           target = std::move(scan);
                         } else if (cnt == 1) {
                           source = std::move(scan);
                         }
                         cnt++;
                         return true;
                       })
      .Process();
  Sophus::SE2d Tts;

  ICP2D::P2PGN(*target, *source, Tts, 10, true);
  LOG(INFO) << "translation: " << Tts.translation().transpose()
            << ", ang: " << Tts.so2().log();
  EXPECT_LE(Tts.translation().norm(), 0.01);
  EXPECT_LE(std::abs(Tts.so2().log()), 0.01);
}

TEST(ICP2DTest, BasicP2PG2o) {
  BagIO bag_io(FLAGS_bag_file);
  size_t cnt = 0;
  std::unique_ptr<sensor_msgs::msg::LaserScan> target;
  std::unique_ptr<sensor_msgs::msg::LaserScan> source;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&cnt, &target, &source](
                           std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         if (cnt == 0) {
                           target = std::move(scan);
                         } else if (cnt == 1) {
                           source = std::move(scan);
                         }
                         cnt++;
                         return true;
                       })
      .Process();
  Sophus::SE2d Tts;

  ICP2D::P2PG2o(*target, *source, Tts, 10, true);
  LOG(INFO) << "translation: " << Tts.translation().transpose()
            << ", ang: " << Tts.so2().log();
  EXPECT_LE(Tts.translation().norm(), 0.01);
  EXPECT_LE(std::abs(Tts.so2().log()), 0.01);
}

TEST(ICP2DTest, BasicP2PCeres) {
  BagIO bag_io(FLAGS_bag_file);
  size_t cnt = 0;
  std::unique_ptr<sensor_msgs::msg::LaserScan> target;
  std::unique_ptr<sensor_msgs::msg::LaserScan> source;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&cnt, &target, &source](
                           std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         if (cnt == 0) {
                           target = std::move(scan);
                         } else if (cnt == 1) {
                           source = std::move(scan);
                         }
                         cnt++;
                         return true;
                       })
      .Process();
  Sophus::SE2d Tts;

  ICP2D::P2PCeres(*target, *source, Tts, 10, true);
  LOG(INFO) << "translation: " << Tts.translation().transpose()
            << ", ang: " << Tts.so2().log();
  EXPECT_LE(Tts.translation().norm(), 0.01);
  EXPECT_LE(std::abs(Tts.so2().log()), 0.1);
}

// TEST(ICP2DTest, G2oVsCeres) {
//   BagIO bag_io(FLAGS_bag_file);
//   std::unique_ptr<sensor_msgs::msg::LaserScan> last_scan;
//   bool initialized = false;
//   double max_translation = 0;
//   double max_angle = 0;
//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&initialized, &last_scan, &max_translation,
//            &max_angle](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             if (!initialized) {
//               last_scan = std::move(scan);
//               initialized = true;
//               return true;
//             }
//             Sophus::SE2d pose_g2o;
//             ICP2D::P2PG2o(*last_scan, *scan, pose_g2o, 10, false);
//             Sophus::SE2d pose_ceres;
//             ICP2D::P2PGN(*last_scan, *scan, pose_ceres, 10, false);
//             const double trans_diff =
//                 (pose_g2o.translation() - pose_ceres.translation()).norm();
//             const double angle_diff =
//                 (pose_g2o.so2().log() - pose_ceres.so2().log());
//             max_translation = std::max(max_translation, trans_diff);
//             max_angle = std::max(max_angle, angle_diff);
//             last_scan = std::move(scan);
//             return true;
//           })
//       .Process();
//   LOG(INFO) << "max translation diff: " << max_translation
//             << ", max angle diff: " << max_angle;
//   SUCCEED();
// }

TEST(ICP2DTest, VisualP2P) {
  BagIO bag_io(FLAGS_bag_file);
  std::unique_ptr<sensor_msgs::msg::LaserScan> last_scan;
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
           &opt_type](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
            if (!initialized) {
              last_scan = std::move(scan);
              initialized = true;
              return true;
            }

            Sophus::SE2d pose;
            ICP2D::P2P(*last_scan, *scan, pose, 10, false, opt_type);
            cv::Mat img;
            Visualizer::Visualize2dScan(*last_scan, Sophus::SE2d(), img,
                                        cv::Vec3b(255, 0, 0), false);
            Visualizer::Visualize2dScan(*scan, pose, img, cv::Vec3b(0, 0, 255),
                                        true);
            cv::imshow("ICP2d: Point to Point", img);
            cv::waitKey(20);
            last_scan = std::move(scan);
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