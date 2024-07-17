#ifndef GENERATE_MASK_HPP
#define GENERATE_MASK_HPP

#include <opencv2/opencv.hpp>
#include <iostream>

void generateMask(std::vector<cv::Vec4f> coord_balls, std::vector<cv::Rect> tracked_balls_bbx, std::vector<std::vector<cv::Point>> corners, int cols, int rows, std::string filename);


#endif 