#include "bag_io.hpp"

void BagIO::Process() {
  rosbag2_storage::StorageOptions options;
  options.uri = bag_file_;
  std::unique_ptr<rosbag2_cpp::Reader> reader =
      rosbag2_transport::ReaderWriterFactory::make_reader(options);
  reader->open(options);

  while (reader->has_next()) {
    rosbag2_storage::SerializedBagMessageConstSharedPtr msg =
        reader->read_next();
    auto func_it = process_funcs_.find(msg->topic_name);
    if (func_it != process_funcs_.end()) {
      func_it->second(msg);
    }
  }
}