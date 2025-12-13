#pragma once
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_transport/reader_writer_factory.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <functional>
#include <map>
#include <string>

class BagIO {
public:
  BagIO(const std::string &bag_file) : bag_file_(bag_file) {}
  using ProcessFunc = std::function<bool(
      rosbag2_storage::SerializedBagMessageConstSharedPtr msg)>;
  using Scan2dHandle = std::function<bool(sensor_msgs::msg::LaserScan)>;

  BagIO &AddHandle(const std::string &topic_name, ProcessFunc process_func) {
    process_funcs_.emplace(topic_name, process_func);
    return *this;
  }
  BagIO &AddScan2dHandle(const std::string &topic_name, Scan2dHandle func) {
    return AddHandle(
        topic_name,
        [&topic_name, func](
            rosbag2_storage::SerializedBagMessageConstSharedPtr msg) -> bool {
          if (msg->topic_name != topic_name) {
            return false;
          }
          rclcpp::SerializedMessage serialized_msg(*msg->serialized_data);
          sensor_msgs::msg::LaserScan::UniquePtr scan_msg =
              std::make_unique<sensor_msgs::msg::LaserScan>();
          rclcpp::Serialization<sensor_msgs::msg::LaserScan> scan_serializer;
          scan_serializer.deserialize_message(&serialized_msg, scan_msg.get());
          if (!scan_msg) {
            return false;
          }
          return func(*scan_msg);
        });
  }
  void Process();

private:
  std::map<std::string, ProcessFunc> process_funcs_;
  std::string bag_file_;
};