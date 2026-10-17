cmake_minimum_required(VERSION 3.16)
project(rls_sync_monitor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -O2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)

add_executable(sync_monitor src/sync_monitor.cpp)
target_link_libraries(sync_monitor PUBLIC
  rclcpp::rclcpp
  ${std_msgs_TARGETS} ${sensor_msgs_TARGETS} ${geometry_msgs_TARGETS} ${diagnostic_msgs_TARGETS})

install(TARGETS sync_monitor DESTINATION lib/${PROJECT_NAME})
ament_package()
