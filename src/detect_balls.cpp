#include "detect_balls.h"


void detectBalls(cv::Mat& src, cv::Mat& output){

    cv::Mat gray;
    cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    imshow("Balls", gray);
    cv::waitKey(0);

    cv::Mat canny;
    Canny(gray,canny,50,100,3);
    cv::imshow("Balls",  canny);
    cv::waitKey(0);

    std::vector<cv::Vec3f> circles;
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows/16, 100, 20, 1, 40);
    for( size_t i = 0; i < circles.size(); i++){
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        circle(output, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
        int radius = c[2];
        circle(output, center, radius, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }

}