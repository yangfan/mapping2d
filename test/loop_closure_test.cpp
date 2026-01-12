// #include "Frame.h"
// #include "LikelihoodFieldMR.h"
#include "Mapper2d.h"
// #include "OGM.h"
#include "bag_io.hpp"
// #include "visualizer.h"

#include <Eigen/Geometry>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sophus/so3.hpp>

// #include <filesystem>
// #include <fstream>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");
DEFINE_string(map_path,
              "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/build/maps/",
              "path to save/load frame");

DEFINE_string(optimizer_type, "G2o", "G2o, Ceres, GN");

// TEST(LC, FrameTest) {
//   BagIO bag_io(FLAGS_bag_file);
//   size_t id = 0;
//   bag_io
//       .AddScan2dHandle(
//           "/pavo_scan_bottom",
//           [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
//             auto frame = std::make_unique<Frame>(std::move(scan));
//             frame->set_kf_id(id++);
//             const std::string file =
//                 FLAGS_map_path + "kf" + std::to_string(id) + ".txt";
//             EXPECT_TRUE(frame->save(file));
//             auto loaded_frame = std::make_unique<Frame>();
//             EXPECT_TRUE(loaded_frame->load(file));

//             EXPECT_EQ(frame->scan().ranges.size(),
//                       loaded_frame->scan().ranges.size());
//             for (size_t i = 0; i < frame->scan().ranges.size(); ++i) {
//               EXPECT_EQ(frame->scan().ranges[i], frame->scan().ranges[i]);
//             }
//             std::filesystem::remove(file);

//             return true;
//           })
//       .Process();
//   SUCCEED();
// }

// TEST(LC, LKMRDist) {
//   cv::Mat grid_map =
//       cv::imread(FLAGS_map_path + "submap0.png", cv::IMREAD_GRAYSCALE);
//   LikelihoodFieldMR lfs(1000, 20);
//   lfs.set_dist_map(grid_map);
//   for (int id = 0; id < 4; ++id) {
//     cv::Mat dismap = lfs.dist_map(id);
//     cv::imshow("dismap" + std::to_string(id), dismap);
//   }
//   cv::waitKey(0);
//   SUCCEED();
// }
// TEST(LC, LFMR) {
//   const std::string sub_info = FLAGS_map_path + "submaps_info.txt";
//   std::ifstream ifs(sub_info);
//   EXPECT_TRUE(ifs.is_open());
//   std::vector<std::unique_ptr<Submap>> subs;
//   subs.reserve(30);
//   LoopClosure lc_detector;
//   int id = 0;
//   double x = 0, y = 0, theta = 0;
//   while (ifs >> id >> x >> y >> theta) {
//     subs.emplace_back(std::make_unique<Submap>(
//         1000, 20, 1, 0.25, Sophus::SE2d(theta, Eigen::Vector2d(x, y)), id));
//     cv::Mat img =
//         cv::imread(FLAGS_map_path + "submap" + std::to_string(id) + ".png",
//                    cv::IMREAD_GRAYSCALE);
//     subs.back()->grid_map().set_map(img);
//     lc_detector.add_submap(subs.back().get());
//   }
//   LOG(INFO) << "Number of submaps: " << subs.size();
//   for (size_t sid = 0; sid < subs.size(); ++sid) {
//     auto &lf = lc_detector.likelihood_mr(sid);
//     for (size_t lid = 0; lid < 4; ++lid) {
//       LOG(INFO) << "submap " << sid << ", level: " << lid;
//       cv::Mat dismap = lf.dist_map(lid);
//       cv::imshow("dismap", dismap);
//       LikelihoodField lfs(1, 1000, 20, 0.25);
//       lfs.set_dist_map(subs[sid]->grid_map());
//       cv::Mat dismap_single = lfs.get_dist_map();
//       cv::imshow("single dist map", dismap_single);
//       cv::waitKey(0);
//     }
//     cv::Mat dismap = lf.dist_map(3);
//     LikelihoodField lfs(1, 1000, 20, 0.25);
//     lfs.set_dist_map(subs[sid]->grid_map());
//     cv::Mat dismap_single = lfs.get_dist_map();

//     EXPECT_EQ(dismap.cols, dismap_single.cols);
//     EXPECT_EQ(dismap.rows, dismap_single.rows);
//     for (int r = 0; r < dismap.rows; ++r) {
//       for (int c = 0; c < dismap.cols; ++c) {
//         EXPECT_EQ(dismap.at<uchar>(r, c), dismap_single.at<uchar>(r, c));
//       }
//     }
//   }
// }

// TEST(LC, M2M) {

//   std::ifstream ifs(FLAGS_map_path + "loop_info.txt");
//   int id1 = 0, id2 = 0;
//   double x = 0.0, y = 0.0, theta = 0.0;
//   ifs >> id1 >> id2 >> x >> y >> theta;

//   auto sub1 = std::make_unique<Submap>(1000, 20, 1, 0.25, Sophus::SE2d(),
//   id1); cv::Mat img1 =
//       cv::imread(FLAGS_map_path + "submap" + std::to_string(id1) + ".png",
//                  cv::IMREAD_GRAYSCALE);
//   sub1->grid_map().set_map(img1);
//   cv::imshow("sub1", img1);

//   auto sub2 = std::make_unique<Submap>(
//       1000, 20, 1, 0.25, Sophus::SE2d(theta, Eigen::Vector2d(x, y)), id2);
//   cv::Mat img2 =
//       cv::imread(FLAGS_map_path + "submap" + std::to_string(id2) + ".png",
//                  cv::IMREAD_GRAYSCALE);
//   sub2->grid_map().set_map(img2);
//   LOG(INFO) << "sub2 Twl: " << sub2->Twl().translation().transpose() << ", "
//             << sub2->Twl().so2().log();
//   cv::imshow("sub2", img2);

//   cv::waitKey(0);

//   std::vector<Submap *> submaps = {sub2.get(), sub1.get()};
//   cv::Mat img = Visualizer::global_map(submaps, 1000);
//   cv::imshow("global", img);
//   cv::waitKey(0);
// }

TEST(LC, LCFrame) {
  auto optimizer_type = LikelihoodField::SolverType::G2o;
  if (FLAGS_optimizer_type == "Ceres") {
    optimizer_type = LikelihoodField::SolverType::Ceres;
  } else if (FLAGS_optimizer_type == "GN") {
    optimizer_type = LikelihoodField::SolverType::GN;
  }

  BagIO bag_io(FLAGS_bag_file);

  Mapper2d mapper;
  // mapper.save_map(FLAGS_map_path, 126, 130);
  mapper.save_map(FLAGS_map_path, 127, 127);
  mapper.set_loop_closure(true, FLAGS_map_path + "loop_info.txt", 1, 15.0,
                          optimizer_type);
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         const bool res =
                             mapper.add_scan(std::move(scan), optimizer_type);
                         mapper.visualize();

                         return res;
                       })
      .Process();
  cv::Mat global_map = mapper.global_map(2000);
  cv::imwrite(FLAGS_map_path + "global_map" + FLAGS_optimizer_type + ".png",
              global_map);
  SUCCEED();
}

// TEST(LC, Aign) {
//   std::shared_ptr<Frame> frame(new Frame());
//   frame->load(FLAGS_map_path + "f919.txt");

//   // std::unique_ptr<Frame> frame = nullptr;
//   // BagIO bag_io(FLAGS_bag_file);
//   // bag_io
//   //     .AddScan2dHandle("/pavo_scan_bottom",
//   //                      [&](std::unique_ptr<sensor_msgs::msg::LaserScan>
//   scan)
//   //                      {
//   //                        if (!frame) {
//   //                          frame =
//   std::make_unique<Frame>(std::move(scan));
//   //                        }
//   //                        return true;
//   //                      })
//   //     .Process();

//   // cv::Mat img_global = cv::imread(FLAGS_map_path + "global_mapG2o.png");
//   cv::Mat img_scan;
//   Visualizer::Visualize2dScan(frame->scan(), Sophus::SE2d(), img_scan,
//                               cv::Vec3b(0, 0, 255), true, 1000);
//   cv::imshow("scan", img_scan);
//   cv::waitKey(0);

//   const std::string sub_info = FLAGS_map_path + "submaps_info.txt";
//   std::ifstream ifs(sub_info);
//   EXPECT_TRUE(ifs.is_open());
//   std::vector<std::unique_ptr<Submap>> subs;
//   subs.reserve(30);
//   LoopClosure lc_detector;
//   int id = 0;
//   double x = 0, y = 0, theta = 0;
//   while (ifs >> id >> x >> y >> theta) {
//     subs.emplace_back(std::make_unique<Submap>(
//         1000, 20, 1, 0.25, Sophus::SE2d(theta, Eigen::Vector2d(x, y)), id));
//     cv::Mat img =
//         cv::imread(FLAGS_map_path + "submap" + std::to_string(id) + ".png",
//                    cv::IMREAD_GRAYSCALE);
//     subs.back()->grid_map().set_map(img);
//     lc_detector.add_submap(subs.back().get());
//   }
//   LOG(INFO) << "Number of submaps: " << subs.size();

//   lc_detector.process(frame.get());
//   for (const auto &[pr, constraint] : lc_detector.loop_constraints()) {
//     cv::Mat img = subs[pr.first]->grid_map().map();
//     Visualizer::Visualize2dScan(frame->scan(), constraint.T_l1_f, img,
//                                 cv::Vec3b(0, 0, 255), true, 1000);
//     cv::imshow("submap" + std::to_string(pr.first), img);
//     cv::waitKey(0);
//   }
// }

// TEST(LC, LCProcess) {

//   BagIO bag_io(FLAGS_bag_file);

//   LoopClosure lc_detector;
//   const std::string sub_info = FLAGS_map_path + "submaps_info.txt";
//   std::ifstream ifs(sub_info);
//   EXPECT_TRUE(ifs.is_open());

//   std::vector<std::unique_ptr<Submap>> subs;
//   subs.reserve(30);

//   while (!ifs.eof()) {
//     int id = 0;
//     double x = 0, y = 0, theta = 0;
//     ifs >> id >> x >> y >> theta;
//     subs.emplace_back(std::make_unique<Submap>(
//         1000, 20, 1, 0.25, Sophus::SE2d(theta, Eigen::Vector2d(x, y)), id));
//     lc_detector.add_submap(subs.back().get());
//   }
//   LOG(INFO) << "Number of submaps: " << subs.size();

//   int cnt = 0;
//   std::set<std::pair<int, int>> visited;
//   bag_io
//       .AddScan2dHandle("/pavo_scan_bottom",
//                        [&](std::unique_ptr<sensor_msgs::msg::LaserScan> scan)
//                        {
//                          auto frame =
//                          std::make_unique<Frame>(std::move(scan)); if
//                          (lc_detector.process(frame.get())) {
//                            for (const auto &[pr, constraint] :
//                                 lc_detector.loop_constraints()) {
//                              if (visited.find(pr) == visited.end()) {
//                                visited.insert(pr);
//                                cv::Mat img =
//                                subs[pr.first]->grid_map().map();
//                                Visualizer::Visualize2dScan(
//                                    frame->scan(), constraint.T_l1_f, img,
//                                    cv::Vec3b(0, 0, 255));
//                              }
//                            }
//                            frame->save(FLAGS_bag_file + "f" +
//                                        std::to_string(cnt++) + ".txt");
//                          }

//                          return true;
//                        })
//       .Process();
//   LOG(INFO) << "cnt: " << cnt;
//   SUCCEED();
// }

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}