#ifndef DETECT_TABLE_HPP
#define DETECT_TABLE_HPP

#include <opencv2/opencv.hpp>

void detectTable(cv::Mat& src, cv::Mat& output);
cv::Mat regionGrowing(const cv::Mat image, cv::Vec3b color, bool start_from_center, bool color_region);

#endif 
