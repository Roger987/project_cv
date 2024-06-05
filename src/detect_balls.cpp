// #include<opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include<opencv2/core.hpp>
// #include<opencv2/imgcodecs.hpp>
// #include <opencv2/video/tracking.hpp>
// #include <iostream>

#include "detect_balls.hpp"

std::vector<cv::Vec3f> detectBalls(cv::Mat& src, cv::Mat& output){

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    // Apply Gaussian blur
    cv::GaussianBlur(src, src, cv::Size(3,3), 1);

    //cv::imshow("Balls", src);
    cv::erode(src, src, 3, cv::Point2d(0,0), 1);
    cv::dilate(src, src, 3, cv::Point2d(0,0), 1);

    std::vector<cv::Vec3f> circles;
    //High threshold to eliminate all apart from strong changes in the color
    HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 5, 700, 10, 7, 17);
    for(int i=0; i<circles.size(); i++)
        std::cout<<circles[i]<<std::endl;

    std::vector<cv::Vec3f> detected_balls;

    for(int i=0; i<circles.size(); i++){
        //c[0] = x coor, c[1] = y coor, c[2] = radius
        cv::Vec3f c = circles[i];
        cv::Rect ball(c[0]-c[2], c[1]-c[2], c[2], c[2]);
        //crop the image
        cv::Mat roi = src(ball);
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(roi, mean, stddev);

        if(0 < mean[0] && mean[0] < 160){
            std::cout<<"Pass"<<mean<<std::endl;
            cv::Point center = cv::Point(c[0], c[1]);
            //circle certer
            circle(output, center, 1, cv::Scalar(0,100,100));
            //circle radius
            float radius = c[2]; 
            cv::Point2i top_left(c[0]-radius, c[1]-radius);
            cv::Point2i bottom_right(c[0]+radius, c[1]+radius);
            cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,255), 1);
            //Save detected balls
            detected_balls.push_back(c);
        }   
    }

    return detected_balls;
}