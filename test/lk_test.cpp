#include "LikelihoodField.h"
#include "OGM.h"
#include "bag_io.hpp"
#include "src/likelihood_field.h"
#include "visualizer.h"

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

TEST(Likelihood, LikelihoodField) {
  BagIO bag_io(FLAGS_bag_file);
  sad::LikelihoodField likelihood_field;
  bool scan_obtained = false;
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&scan_obtained, &likelihood_field](
                           std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         if (!scan_obtained) {
                           likelihood_field.SetTargetScan(std::move(scan));
                           scan_obtained = true;
                         }
                         return true;
                       })
      .Process();
  cv::Mat img = likelihood_field.GetFieldImage();
  cv::imwrite(FLAGS_likelihood_field_file, img);
  SUCCEED();
}

// TEST(Likelihood, ALignTestBasic) {
//   BagIO bag_io(FLAGS_bag_file);
//   size_t cnt = 0;
//   std::unique_ptr<sensor_msgs::msg::LaserScan> target;
//   std::unique_ptr<sensor_msgs::msg::LaserScan> source;
//   bag_io
//       .AddScan2dHandle("/pavo_scan_bottom",
//                        [&cnt, &target, &source](
//                            std::unique_ptr<sensor_msgs::msg::LaserScan> scan)
//                            {
//                          if (cnt == 0) {
//                            target = std::move(scan);
//                          } else if (cnt == 1) {
//                            source = std::move(scan);
//                          }
//                          cnt++;
//                          return true;
//                        })
//       .Process();
//   LikelihoodField lf(1, 1000, 20, 0.25);
//   lf.set_dist_map(*target);
//   Sophus::SE2d pose;
//   EXPECT_TRUE(lf.align_test(*source, pose, 10, true));
//   LOG(INFO) << "translation: " << pose.translation().transpose()
//             << ", angle: " << pose.so2().log();
//   cv::Mat dist_map = lf.get_dist_map();
//   cv::imshow("dist map", dist_map);
//   cv::waitKey(0);
//   SUCCEED();
// }

TEST(GridMap, LocalMappingLFRef) {
  OccupancyGridMap grid_map(1000, 20);
  BagIO bag_io(FLAGS_bag_file);
  LikelihoodField lf(1, 1000, 20, 0.25);
  std::shared_ptr<Frame> last_frame = nullptr;
  std::shared_ptr<Frame> last_keyframe = nullptr;
  Sophus::SE2d guess{};
  Sophus::SE2d sub_pose{};

  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
            auto frame = std::make_shared<Frame>(std::move(scan));
            if (!last_frame) {
              grid_map.add_frame(frame->scan(), Sophus::SE2d());
              lf.set_dist_map(grid_map);
              last_frame = frame;
              last_keyframe = frame;
              return true;
            } else {
              frame->set_Twf(last_frame->Twf() * guess);
              frame->set_Tlf(last_frame->Tlf());
            }
            Sophus::SE2d initial_pose = frame->Tlf();
            lf.scan2map_G2o(frame->scan(), initial_pose, 10, false);
            // lf.align_test(frame->scan(), initial_pose, 10, false);
            frame->set_Tlf(initial_pose);
            frame->set_Twf(sub_pose * frame->Tlf());

            Sophus::SE2d rel = last_keyframe->Twf().inverse() * frame->Twf();
            if (rel.translation().norm() > 0.3 ||
                rel.so2().log() > 15 * M_PI / 180) {
              grid_map.add_frame(frame->scan(), frame->Twf());
              //   lf.SetFieldImageFromOccuMap(grid_map.map());
              lf.set_dist_map(grid_map);
              last_keyframe = frame;
              LOG(INFO) << "Add.";
            }
            cv::Mat img = grid_map.binary_map(127, 127);
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

            return true;
          })
      .Process();
}

// TEST(GridMap, LocalMappingFrame) {
//   OccupancyGridMap grid_map(1000, 20);
//   BagIO bag_io(FLAGS_bag_file);
//   sad::LikelihoodField lf;
//   std::shared_ptr<Frame> last_frame = nullptr;
//   std::shared_ptr<Frame> last_keyframe = nullptr;
//   Sophus::SE2d guess{};
//   Sophus::SE2d sub_pose{};

//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             auto frame = std::make_shared<Frame>(std::move(scan));
//             if (!last_frame) {
//               grid_map.add_frame(frame->scan(), Sophus::SE2d());
//               lf.SetFieldImageFromOccuMap(grid_map.map());
//               last_frame = frame;
//               last_keyframe = frame;
//               return true;
//             } else {
//               frame->set_Twf(last_frame->Twf() * guess);
//               frame->set_Tlf(last_frame->Tlf());
//             }
//             lf.SetSourceScan(
//                 std::make_shared<sensor_msgs::msg::LaserScan>(frame->scan()));
//             Sophus::SE2d initial_pose = frame->Tlf();
//             lf.AlignG2O(initial_pose);
//             frame->set_Tlf(initial_pose);
//             frame->set_Twf(sub_pose * frame->Tlf());

//             Sophus::SE2d rel = last_keyframe->Twf().inverse() * frame->Twf();
//             if (rel.translation().norm() > 0.3 ||
//                 rel.so2().log() > 15 * M_PI / 180) {
//               grid_map.add_frame(frame->scan(), frame->Twf());
//               lf.SetFieldImageFromOccuMap(grid_map.map());
//               last_keyframe = frame;
//               LOG(INFO) << "Add.";
//             }
//             cv::Mat img = grid_map.binary_map(127, 127);
//             Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("grid map", img);

//             cv::Mat lf_img = lf.GetFieldImage();
//             Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), lf_img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("likelihood field", lf_img);
//             cv::waitKey(20);

//             guess = last_frame->Twf().inverse() * frame->Twf();
//             last_frame = frame;

//             return true;
//           })
//       .Process();
// }

// TEST(GridMap, LocalMappingOGM) {
//   OccupancyGridMap grid_map(1000, 20);
//   BagIO bag_io(FLAGS_bag_file);
//   sad::LikelihoodField lf;
//   std::shared_ptr<sad::Frame> last_frame = nullptr;
//   std::shared_ptr<sad::Frame> last_keyframe = nullptr;
//   Sophus::SE2d guess{};
//   Sophus::SE2d sub_pose{};

//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             auto frame = std::make_shared<sad::Frame>(
//                 std::make_shared<sensor_msgs::msg::LaserScan>(*scan));
//             if (!last_frame) {
//               grid_map.add_frame(*scan, Sophus::SE2d());
//               lf.SetFieldImageFromOccuMap(grid_map.map());
//               last_frame = frame;
//               last_keyframe = frame;
//               return true;
//             } else {
//               frame->pose_ = last_frame->pose_ * guess;
//               frame->pose_submap_ = last_frame->pose_submap_;
//             }
//             lf.SetSourceScan(
//                 std::make_shared<sensor_msgs::msg::LaserScan>(*scan));
//             lf.AlignG2O(frame->pose_submap_);
//             frame->pose_ = sub_pose * frame->pose_submap_;
//             Sophus::SE2d rel = last_keyframe->pose_.inverse() * frame->pose_;
//             if (rel.translation().norm() > 0.3 ||
//                 rel.so2().log() > 15 * M_PI / 180) {
//               grid_map.add_frame(*scan, frame->pose_);
//               lf.SetFieldImageFromOccuMap(grid_map.map());
//               last_keyframe = frame;
//               LOG(INFO) << "Add.";
//             }
//             cv::Mat img = grid_map.binary_map(127, 127);
//             Visualizer::Visualize2dScan(*scan, frame->pose_, img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("grid map", img);

//             cv::Mat lf_img = lf.GetFieldImage();
//             Visualizer::Visualize2dScan(*scan, frame->pose_, lf_img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("likelihood field", lf_img);
//             cv::waitKey(20);

//             guess = last_frame->pose_.inverse() * frame->pose_;
//             last_frame = frame;

//             return true;
//           })
//       .Process();
// }

// TEST(GridMap, LocalMappingRef) {
//   sad::OccupancyMap grid_map;
//   BagIO bag_io(FLAGS_bag_file);
//   sad::LikelihoodField lf;
//   std::shared_ptr<sad::Frame> last_frame = nullptr;
//   std::shared_ptr<sad::Frame> last_keyframe = nullptr;
//   Sophus::SE2d guess{};
//   Sophus::SE2d sub_pose{};

//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             auto frame = std::make_shared<sad::Frame>(
//                 std::make_shared<sensor_msgs::msg::LaserScan>(*scan));
//             if (!last_frame) {
//               grid_map.AddLidarFrame(frame);
//               lf.SetFieldImageFromOccuMap(grid_map.GetOccupancyGrid());
//               last_frame = frame;
//               last_keyframe = frame;
//               return true;
//             } else {
//               frame->pose_ = last_frame->pose_ * guess;
//               frame->pose_submap_ = last_frame->pose_submap_;
//             }
//             lf.SetSourceScan(
//                 std::make_shared<sensor_msgs::msg::LaserScan>(*scan));
//             lf.AlignG2O(frame->pose_submap_);
//             frame->pose_ = sub_pose * frame->pose_submap_;
//             Sophus::SE2d rel = last_keyframe->pose_.inverse() * frame->pose_;
//             if (rel.translation().norm() > 0.3 ||
//                 rel.so2().log() > 15 * M_PI / 180) {
//               grid_map.AddLidarFrame(frame);
//               lf.SetFieldImageFromOccuMap(grid_map.GetOccupancyGrid());
//               last_keyframe = frame;
//               LOG(INFO) << "Add.";
//             }
//             cv::Mat img = grid_map.GetOccupancyGridBlackWhite();
//             Visualizer::Visualize2dScan(*scan, frame->pose_, img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("grid map", img);

//             cv::Mat lf_img = lf.GetFieldImage();
//             Visualizer::Visualize2dScan(*scan, frame->pose_, lf_img,
//                                         cv::Vec3b(0, 0, 255), true, 1000);
//             cv::imshow("likelihood field", lf_img);
//             cv::waitKey(20);

//             guess = last_frame->pose_.inverse() * frame->pose_;
//             last_frame = frame;

//             return true;
//           })
//       .Process();
// }

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}