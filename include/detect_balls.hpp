#ifndef DETECT_BALLS_HPP
#define DETECT_BALLS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


std::vector<cv::Vec3f> detectBalls(cv::Mat& src, cv::Mat& output, int segmentation);

void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>>& white_balls, int& segmentation, cv::Mat& img, cv::Mat& output);

void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>>& solid_balls, int& segmentation, cv::Mat& img, cv::Mat& output);

#endif 