#ifndef DETECT_AND_CLASSIFY_BALLS_HPP
#define DETECT_AND_CLASSIFY_BALLS_HPP

//Author: Francesco Stella

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>


void detectAndClassifyBalls(cv::Mat& src, cv::Mat& output, int segmentation, std::vector<cv::Vec4f>& classified_balls);

void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& white_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls);

void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& solid_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls);

// Compute the entropy of the provided image. Used to detect false posivites balls
double calculateEntropy(const cv::Mat& histogram);

// Compute the histogram of an image and return its class
int histogramCal(cv::Mat img);
#endif 