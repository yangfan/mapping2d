#include "bag_io.hpp"
#include "visualizer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/2dmapping/ros2_f1",
              "path of ros2 bag file");

TEST(UtilsTest, VisualizerIO) {
  BagIO bag_io(FLAGS_bag_file);
  bag_io
      .AddScan2dHandle("/pavo_scan_bottom",
                       [](std::unique_ptr<sensor_msgs::msg::LaserScan> scan) {
                         cv::Mat img;
                         Visualizer::Visualize2dScan(*scan, Sophus::SE2d(), img,
                                                     cv::Vec3b(0, 0, 255));
                         cv::imshow("scan", img);
                         cv::waitKey(20);
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