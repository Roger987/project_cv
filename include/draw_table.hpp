#ifndef DRAW_TABLE_HPP
#define DRAW_TABLE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


cv::Mat drawTable(std::vector<cv::Vec4f> coord_balls, cv::Mat M);


#endif 