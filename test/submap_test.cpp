#include "LikelihoodField.h"
#include "Submap.h"
#include "bag_io.hpp"
#include "visualizer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sophus/se2.hpp>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");

DEFINE_string(submap_path,
              "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/build/submap",
              "path of submap file");

DEFINE_string(optimizer_type, "G2o", "G2o, Ceres, GN");

// TEST(SubMap, LocalMapping) {

//   auto optimizer_type = LikelihoodField::SolverType::G2o;
//   if (FLAGS_optimizer_type == "Ceres") {
//     optimizer_type = LikelihoodField::SolverType::Ceres;
//   } else if (FLAGS_optimizer_type == "GN") {
//     optimizer_type = LikelihoodField::SolverType::GN;
//   }

//   Submap submap(1000, 20, 1, 0.25, Sophus::SE2d(), 0);
//   std::shared_ptr<Frame> last_frame = nullptr;
//   std::shared_ptr<Frame> last_keyframe = nullptr;
//   Sophus::SE2d last_motion;
//   constexpr double kmin_translation = 0.3;
//   constexpr double kmin_rotation = 15 * M_PI / 180;

//   BagIO bag_io(FLAGS_bag_file);
//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             auto frame = std::make_shared<Frame>(std::move(scan));
//             if (!last_frame) {
//               submap.initialize(frame);
//               last_frame = frame;
//               last_keyframe = frame;
//               return true;
//             }
//             if (submap.has_outsider()) {
//               return true;
//             }
//             frame->set_Tlf(last_frame->Tlf());
//             frame->set_Twf(last_frame->Twf() * last_motion);

//             if (!submap.scan_match(*frame, optimizer_type)) {
//               return false;
//             }
//             Sophus::SE2d rel_motion =
//                 last_keyframe->Twf().inverse() * frame->Twf();
//             if (rel_motion.translation().norm() > kmin_translation ||
//                 std::abs(rel_motion.so2().log()) > kmin_rotation) {
//               submap.add_keyframe(frame);
//               last_keyframe = frame;
//             }

//             last_motion = last_frame->Twf().inverse() * frame->Twf();
//             last_frame = frame;

//             cv::Mat grid_map = submap.grid_map().binary_map(126, 130);
//             Visualizer::Visualize2dScan(frame->scan(), frame->Twf(),
//             grid_map,
//                                         cv::Vec3b(0, 0, 255), true, 1000, 20,
//                                         submap.Twl());
//             cv::imshow("grid map", grid_map);

//             cv::Mat dist_map = submap.dist_map();
//             Visualizer::Visualize2dScan(frame->scan(), frame->Twf(),
//             dist_map,
//                                         cv::Vec3b(0, 0, 255), true, 1000, 20,
//                                         submap.Twl());
//             cv::imshow("distance map", dist_map);

//             cv::waitKey(20);

//             if (submap.has_outsider()) {
//               cv::imwrite(FLAGS_submap_path + FLAGS_optimizer_type + ".png",
//                           submap.grid_map().binary_map(126, 130));
//             }

//             return true;
//           })
//       .Process();
//   SUCCEED();
// }

TEST(SubMap, ExtendMap) {

  auto optimizer_type = LikelihoodField::SolverType::G2o;
  if (FLAGS_optimizer_type == "Ceres") {
    optimizer_type = LikelihoodField::SolverType::Ceres;
  } else if (FLAGS_optimizer_type == "GN") {
    optimizer_type = LikelihoodField::SolverType::GN;
  }

  auto submap = std::make_shared<Submap>(1000, 20, 1, 0.25, Sophus::SE2d(), 0);
  std::shared_ptr<Frame> last_frame = nullptr;
  std::shared_ptr<Frame> last_keyframe = nullptr;
  Sophus::SE2d last_motion;
  constexpr double kmin_translation = 0.3;
  constexpr double kmin_rotation = 15 * M_PI / 180;

  BagIO bag_io(FLAGS_bag_file);
  bag_io
      .AddScan2dHandle(
          "/pavo_scan_bottom",
          [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
            auto frame = std::make_shared<Frame>(std::move(scan));
            if (!last_frame) {
              submap->initialize(frame);
              last_frame = frame;
              last_keyframe = frame;
              return true;
            }
            frame->set_Tlf(last_frame->Tlf());
            frame->set_Twf(last_frame->Twf() * last_motion);

            if (!submap->scan_match(*frame, optimizer_type)) {
              return false;
            }
            Sophus::SE2d rel_motion =
                last_keyframe->Twf().inverse() * frame->Twf();
            if (rel_motion.translation().norm() > kmin_translation ||
                std::abs(rel_motion.so2().log()) > kmin_rotation) {
              submap->add_keyframe(frame);
              last_keyframe = frame;
            }

            last_motion = last_frame->Twf().inverse() * frame->Twf();
            last_frame = frame;

            if (submap->has_outsider()) {

              cv::imwrite(FLAGS_submap_path + FLAGS_optimizer_type +
                              std::to_string(submap->id()) + ".png",
                          submap->grid_map().binary_map(126, 130));

              auto last_submap = submap;
              submap = std::make_shared<Submap>(
                  1000, 20, 1, 0.25, last_frame->Twf(), last_submap->id() + 1);
              submap->copy_frames(*last_submap, 10);
              LOG(INFO) << "Extend to Submap " << submap->id();
            }

            cv::Mat grid_map = submap->grid_map().binary_map(126, 130);
            Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), grid_map,
                                        cv::Vec3b(0, 0, 255), true, 1000, 20,
                                        submap->Twl());
            cv::imshow("submap", grid_map);

            cv::Mat dist_map = submap->dist_map();
            Visualizer::Visualize2dScan(frame->scan(), frame->Twf(), dist_map,
                                        cv::Vec3b(0, 0, 255), true, 1000, 20,
                                        submap->Twl());
            cv::imshow("distance map", dist_map);

            cv::waitKey(20);

            return true;
          })
      .Process();
  SUCCEED();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}