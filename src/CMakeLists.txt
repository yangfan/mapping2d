add_library(bag_io bag_io.cpp)
target_link_libraries(bag_io PUBLIC rclcpp::rclcpp rosbag2_transport::rosbag2_transport ${sensor_msgs_TARGETS})
target_include_directories(bag_io PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_library(visualizer visualizer.cpp)
target_link_libraries(visualizer PUBLIC Eigen3::Eigen Sophus::Sophus ${OpenCV_LIBS} ${sensor_msgs_TARGETS})
target_include_directories(visualizer PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

set(G2O_LIBS fmt ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION})
add_library(icp2d icp2d.cpp g2o_types.cpp)
target_link_libraries(icp2d PUBLIC Eigen3::Eigen Sophus::Sophus ${sensor_msgs_TARGETS} ${G2O_LIBS} ${CERES_LIBRARIES})
target_include_directories(icp2d PUBLIC
    ${CMAKE_SOURCE_DIR}/include/mapping2d
    ${G2O_INCLUDE_DIR}
)
