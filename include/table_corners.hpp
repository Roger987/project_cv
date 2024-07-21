#ifndef TABLE_CONTOURS_HPP
#define TABLE_CONTOURS_HPP

#include <opencv2/opencv.hpp>

std::vector<std::vector<cv::Point>> tableCorners(cv::Mat& src);

#endif 
