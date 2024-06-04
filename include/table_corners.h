#ifndef TABLE_CONTOURS_H
#define TABLE_CONTOURS_H

#include <opencv2/opencv.hpp>

std::vector<std::vector<cv::Point>> tableCorners(cv::Mat& src);

#endif 
