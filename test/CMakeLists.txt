enable_testing()

add_executable(utils_test utils_test.cpp)
target_link_libraries(utils_test PUBLIC gtest glog::glog ${GFLAGS_LIBRARIES} bag_io visualizer)
target_include_directories(utils_test PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(kd_test kd_test.cpp)
target_link_libraries(kd_test PUBLIC gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)
target_include_directories(kd_test PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(icp2d_test icp2d_test.cpp)
target_link_libraries(icp2d_test PUBLIC gtest glog::glog bag_io visualizer icp2d ${GFLAGS_LIBRARIES})
target_include_directories(icp2d_test PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(icp2dp2l_test icp2dp2l_test.cpp)
target_link_libraries(icp2dp2l_test PUBLIC gtest glog::glog bag_io visualizer icp2d ${GFLAGS_LIBRARIES})
target_include_directories(icp2dp2l_test PUBLIC ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(likelihood_test likelihood_test.cpp)
target_link_libraries(likelihood_test PRIVATE gtest glog::glog ${GFLAGS_LIBRARIES} likelihood_field bag_io visualizer)
target_include_directories(likelihood_test PRIVATE ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(grid_map_test gridmap_test.cpp)
target_link_libraries(grid_map_test PRIVATE gtest glog::glog ${GFLAGS_LIBRARIES} bag_io visualizer likelihood_field Eigen3::Eigen Sophus::Sophus map)
target_include_directories(grid_map_test PRIVATE ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(submap_test submap_test.cpp)
target_link_libraries(submap_test PRIVATE gtest glog::glog ${GFLAGS_LIBRARIES} bag_io visualizer Eigen3::Eigen Sophus::Sophus map)
target_include_directories(submap_test PRIVATE ${CMAKE_SOURCE_DIR}/include/mapping2d)

add_executable(mapper_test mapper_test.cpp)
target_link_libraries(mapper_test PRIVATE gtest glog::glog ${GFLAGS_LIBRARIES} bag_io visualizer Eigen3::Eigen Sophus::Sophus map)
target_include_directories(mapper_test PRIVATE ${CMAKE_SOURCE_DIR}/include/mapping2d)

include(GoogleTest)
gtest_discover_tests(utils_test kd_test icp2d_test)