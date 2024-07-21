cmake_minimum_required(VERSION 3.8)

project(project_cv VERSION 1.0 LANGUAGES CXX)

# Set the path to OpenCV extra modules
set(OPENCV_EXTRA_MODULES_PATH ../opencv_contrib/modules)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp 
                    src/detect_table.cpp 
                    src/table_corners.cpp 
                    src/detect_contours.cpp 
                    src/detect_and_classify_balls.cpp 
                    src/find_perspective.cpp
                    src/draw_table.cpp
                    src/generate_coords.cpp
                    src/model_evaluation.cpp
                    src/generate_mask.cpp)

target_link_libraries(main ${OpenCV_LIBS})