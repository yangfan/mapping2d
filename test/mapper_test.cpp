#include "LikelihoodField.h"
#include "Mapper2d.h"
#include "bag_io.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");

DEFINE_string(map_path,
              "/home/fan/ssd/Projects/ros2_ws/src/mapping2d/build/maps/",
              "path of submap file");

DEFINE_string(optimizer_type, "G2o", "G2o, Ceres, GN");

TEST(Mapper, Process) {
  auto optimizer_type = LikelihoodField::SolverType::G2o;
  if (FLAGS_optimizer_type == "Ceres") {
    optimizer_type = LikelihoodField::SolverType::Ceres;
  } else if (FLAGS_optimizer_type == "GN") {
    optimizer_type = LikelihoodField::SolverType::GN;
  }

  BagIO bag_io(FLAGS_bag_file);

  Mapper2d mapper;
  mapper.save_map(FLAGS_map_path, 126, 130);
  int cnt = 0;
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
  cv::imwrite(FLAGS_map_path + "global_map.png", global_map);
  LOG(INFO) << "cnt: " << cnt;
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