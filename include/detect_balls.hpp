#ifndef DETECT_BALLS_HPP
#define DETECT_BALLS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


void detectBalls(cv::Mat& src, cv::Mat& output, int segmentation, std::vector<cv::Vec4f>& classified_balls);

void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& white_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls);

void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& solid_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls);

#endif 