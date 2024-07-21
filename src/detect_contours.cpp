//Roger De Almeida Matos Junior

#include "detect_contours.h"

std::vector<std::vector<cv::Point>> detectContours(int rows, int cols, std::vector<std::vector<cv::Point>> corners){
    
    cv::Mat table(rows, cols, CV_8UC1, cv::Scalar(0));
    fillPoly(table, corners, cv::Scalar(255));
    
    cv::Mat table_cont(rows, cols, CV_8UC3, cv::Scalar(0,0,0));
    std::vector<std::vector<cv::Point>> contours;
    findContours(table, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    return contours;
}
