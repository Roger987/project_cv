#ifndef DRAW_TABLE_HPP
#define DRAW_TABLE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


cv::Mat drawTable(std::vector<cv::Rect> tracked_balls_bbx, std::vector<cv::Vec4f> coord_balls, std::vector<std::vector<cv::Point2f>>& trajectories, cv::Mat M);


#endif 