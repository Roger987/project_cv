// #include<opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include<opencv2/core.hpp>
// #include<opencv2/imgcodecs.hpp>
// #include <opencv2/video/tracking.hpp>
// #include <iostream>

#include "detect_balls.hpp"

std::vector<cv::Vec3f> detectBalls(cv::Mat& src){

    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    // Apply Gaussian blur
    cv::GaussianBlur(src, src, cv::Size(11,11), 1, 1);
    cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // contains coordinates of the center and the radius
    std::vector<cv::Vec3f> circles;
    HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 40, 1000, 10, 7, 16);

    std::vector<cv::Vec3f> detected_balls;

    for(int i=0; i<circles.size(); i++){
        //c[0] = center_x coor, c[1] = center_y coor, c[2] = radius
        cv::Vec3f c = circles[i];
        float radius = c[2]; 
        cv::Rect ball_bbox(c[0]-c[2], c[1]-c[2], radius*2, radius*2);
        //crop the image
        cv::Mat roi = src(ball_bbox);
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(roi, mean, stddev);

        //Balls filter
        if(0 < mean[0] && mean[0] < 160){
            //Save detected ball
            detected_balls.push_back(c);
        }   
    }

    //Delete balls when center are close
    //delete_redundant_balls(detected_balls);

    return detected_balls;
}

// void delete_redundant_balls(std::vector<cv::Vec3f>& detected_balls){
//     std::vector<uchar> del_balls;

//     for(int i=0; i<detected_balls.size(); i++){
//         for(int j=i+1; j<detected_balls.size(); j++){
//             if(abs(detected_balls[i][0]-detected_balls[j][0]) < 10 && abs(detected_balls[i][1]-detected_balls[j][1]) < 10)
//                 del_balls.push_back(i);
//         }
//     }

//     for(int i=del_balls.size()-1; i>=0; i--)
//         detected_balls.erase(detected_balls.begin() + del_balls[i]);
// }


void print_bbox(cv::Vec3f& c, cv::Mat& output){
    int radius = c[2];
    cv::Point center = cv::Point(c[0], c[1]);
    //circle certer
    circle(output, center, 1, cv::Scalar(0,100,100));
    //circle radius
    cv::Point2i top_left(c[0]-radius, c[1]-radius);
    cv::Point2i bottom_right(c[0]+radius, c[1]+radius);
    cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,255), 1);
}