add_library(bag_io bag_io.cpp)
target_link_libraries(bag_io PUBLIC rclcpp::rclcpp rosbag2_transport::rosbag2_transport ${sensor_msgs_TARGETS})
target_include_directories(bag_io PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_library(visualizer visualizer.cpp)
target_link_libraries(visualizer PUBLIC Eigen3::Eigen Sophus::Sophus ${OpenCV_LIBS} ${sensor_msgs_TARGETS})
target_include_directories(visualizer PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

set(G2O_LIBS fmt ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION})
add_library(g2o_types g2o_types.cpp)
target_link_libraries(g2o_types PUBLIC TBB::tbb Eigen3::Eigen Sophus::Sophus ${sensor_msgs_TARGETS} ${G2O_LIBS} ${OpenCV_LIBRARIES})
target_include_directories(g2o_types PUBLIC
    ${CMAKE_SOURCE_DIR}/include/mapping2d
    ${G2O_INCLUDE_DIR}
)

add_library(ceres_types ceres_types.cpp)
target_link_libraries(ceres_types PUBLIC ${CERES_LIBRARIES} TBB::tbb Eigen3::Eigen Sophus::Sophus ${sensor_msgs_TARGETS} ${OpenCV_LIBRARIES})
target_include_directories(ceres_types PUBLIC
    ${CMAKE_SOURCE_DIR}/include/mapping2d
)

add_library(icp2d icp2d.cpp)
target_link_libraries(icp2d PUBLIC g2o_types ceres_types TBB::tbb Eigen3::Eigen Sophus::Sophus ${sensor_msgs_TARGETS} ${G2O_LIBS} ${CERES_LIBRARIES})
target_include_directories(icp2d PUBLIC
    ${CMAKE_SOURCE_DIR}/include/mapping2d
    ${G2O_INCLUDE_DIR}
)

add_library(likelihood_field LikelihoodField.cpp)
target_link_libraries(likelihood_field PUBLIC ${OpenCV_LIBRARIES} Eigen3::Eigen Sophus::Sophus ${sensor_msgs_TARGETS} ${G2O_LIBS} g2o_types ceres_types)
target_include_directories(likelihood_field PUBLIC
    ${CMAKE_SOURCE_DIR}/include/mapping2d
    ${CMAKE_SOURCE_DIR}/test/src
)

add_library(map OGM.cpp Submap.cpp Mapper2d.cpp)
target_link_libraries(map PUBLIC ${OpenCV_LIBRARIES} Eigen3::Eigen Sophus::Sophus likelihood_field visualizer ${sensor_msgs_TARGETS})
target_include_directories(map PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)
