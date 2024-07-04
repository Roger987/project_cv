cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(project_cv VERSION 1.0 LANGUAGES CXX)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/detect_table.cpp src/table_corners.cpp src/detect_contours.cpp src/detect_balls.cpp src/find_perspective.cpp)

target_link_libraries(main ${OpenCV_LIBS})