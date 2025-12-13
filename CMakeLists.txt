cmake_minimum_required(VERSION 3.20)
project(mapping2d)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_BUILD_TYPE "Release")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_transport REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(Ceres REQUIRED)

set(G2O_ROOT "/home/fan/ssd/study/slambook2/3rdparty/g2o/install")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
Find_Package(G2O REQUIRED)
message(STATUS "g2o include: ${G2O_INCLUDE_DIR}")
message(STATUS "g2o core: ${G2O_CORE_LIBRARY}")

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)

  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
add_subdirectory(test)

# install(TARGETS
# DESTINATION lib/${PROJECT_NAME}
# )
ament_package()
