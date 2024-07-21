#ifndef DETECT_BALLS_HPP
#define DETECT_BALLS_HPP

#include <opencv2/opencv.hpp>

std::vector<std::vector<cv::Point>> detectContours(int rows, int cols, std::vector<std::vector<cv::Point>> corners);

#endif 
