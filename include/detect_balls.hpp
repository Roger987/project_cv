#ifndef DETECT_BALLS_HPP
#define DETECT_BALLS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


std::vector<cv::Vec3f> detectBalls(cv::Mat& src, cv::Mat& output);

void delete_redundant_balls(std::vector<cv::Vec3f>& obj);

void print_bbox(cv::Vec3f& obj, cv::Mat& output);

#endif 