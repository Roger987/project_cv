#include "generate_mask.hpp"
#include <fstream>
#include <iostream>

void generateMask(std::vector<cv::Vec4f> coord_balls, std::vector<cv::Rect> tracked_balls_bbx, std::vector<std::vector<cv::Point>> corners, int cols, int rows, std::string filename) {


    cv::Mat mask(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::fillPoly(mask, corners, cv::Scalar(5, 5, 5));

    for (size_t i = 0; i < tracked_balls_bbx.size(); i++) {
        int centerX = tracked_balls_bbx[i].x + tracked_balls_bbx[i].width / 2;
        int centerY = tracked_balls_bbx[i].y + tracked_balls_bbx[i].height / 2;
        cv::Scalar color(coord_balls[i][3], coord_balls[i][3], coord_balls[i][3]);
        cv::circle(mask, cv::Point(centerX, centerY), 12, cv::Scalar(1, 1, 1), cv::FILLED);
    }


    cv::imwrite(filename, mask);
    
    return;

}