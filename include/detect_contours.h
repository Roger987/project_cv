#ifndef DETECT_BALLS_H
#define DETECT_BALLS_H

#include <opencv2/opencv.hpp>

std::vector<std::vector<cv::Point>> detectContours(int rows, int cols, std::vector<std::vector<cv::Point>> corners);

#endif 
