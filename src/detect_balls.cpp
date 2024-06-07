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

    cv::Mat del_clone = output.clone();

    for(int i=0; i<circles.size(); i++){
        //c[0] = x coor, c[1] = y coor, c[2] = radius
        cv::Vec3f c = circles[i];
        float radius = c[2]; 
        cv::Rect ball_bbox(c[0]-c[2], c[1]-c[2], radius*2, radius*2);
        //crop the image
        cv::Mat roi = src(ball_bbox);
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(roi, mean, stddev);

        if(0 < mean[0] && mean[0] < 160){
            // std::cout<<c<<std::endl;
            // std::cout<<"Pass"<<mean<<std::endl;
            cv::Point center = cv::Point(c[0], c[1]);
            //circle certer
            circle(output, center, 1, cv::Scalar(0,100,100));
            //circle radius
            cv::Point2i top_left(c[0]-radius, c[1]-radius);
            cv::Point2i bottom_right(c[0]+radius, c[1]+radius);
            cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,255), 1);
            //Save detected balls
            detected_balls.push_back(c);
        }   
    }

    std::cout<<"Before"<<std::endl;
    std::cout<<detected_balls.size()<<std::endl;
    cv::imshow("Detected balls", output);

    delete_redundant_balls(detected_balls);

    std::cout<<"Not deleted"<<std::endl;
    std::cout<<detected_balls.size()<<std::endl;
    for(int i=0; i<detected_balls.size(); i++){
        print_bbox(detected_balls[i], del_clone);
    }

    cv::imshow("Del clone", del_clone);

    cv::waitKey(0);

    return detected_balls;
}


void delete_redundant_balls(std::vector<cv::Vec3f>& detected_balls){

    std::vector<uchar> del_balls;

    for(int i=0; i<detected_balls.size(); i++){
        for(int j=i+1; j<detected_balls.size(); j++){
            if(abs(detected_balls[i][0]-detected_balls[j][0]) < 10 && abs(detected_balls[i][1]-detected_balls[j][1]) < 10)
                del_balls.push_back(i);
        }
    }

    for(int i=del_balls.size()-1; i>=0; i--)
        detected_balls.erase(detected_balls.begin() + del_balls[i]);
    std::cout<<"Del balls"<<std::endl;
}


void print_bbox(cv::Vec3f& c, cv::Mat& output){
    
    std::cout<<c<<std::endl;

    int radius = c[2];
    cv::Point center = cv::Point(c[0], c[1]);
    //circle certer
    circle(output, center, 1, cv::Scalar(0,100,100));
    //circle radius
    cv::Point2i top_left(c[0]-radius, c[1]-radius);
    cv::Point2i bottom_right(c[0]+radius, c[1]+radius);
    cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,255), 1);
}