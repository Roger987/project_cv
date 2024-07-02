// #include<opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include<opencv2/core.hpp>
// #include<opencv2/imgcodecs.hpp>
// #include <opencv2/video/tracking.hpp>
// #include <iostream>

#include "detect_balls.hpp"

double calculateEntropy(const cv::Mat& histogram) {
    double entropy = 0.0;
    double total_pixels = cv::sum(histogram)[0];

    for (int i = 0; i < histogram.rows; ++i) {
        double p = histogram.at<float>(i) / total_pixels;
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}

int histogram_cal(cv::Mat img){
    // cv::imshow("ball", img);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
    // cv::imshow("ball1", thresh);
    // cv::waitKey(0);

    int whitePixelCount = cv::countNonZero(thresh);
    // int maxValue = 255; // Assuming it's a grayscale image
    // for (int y = 0; y < thresh.rows; ++y) {
    //     for (int x = 0; x < thresh.cols; ++x) {
    //         if (thresh.at<uchar>(y, x) == maxValue) {
    //             whitePixelCount++;
    //         }
    //     }
    // }
    int totalPixels = img.rows * img.cols;

    // Calcular o percentual de pixels brancos
    double whitePercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100;

    // Print the result
    if (whitePercentage > 15){
        std::cout << "Number of White pixels: " << whitePixelCount << " " << whitePercentage << std::endl;
        cv::imshow("ball1", img);
        //cv::waitKey(0);
    }
    // std::cout << "Number of White pixels: " << whitePixelCount << " " << whitePercentage << std::endl;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    cv::Mat hist;
    cv::calcHist( &gray, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);

    //int hist_w = 512, hist_h = 400;
    int hist_w = 256, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    cv::Mat histImage( hist_h, hist_w, CV_8UC1, cv::Scalar(0) );

    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 0; i < histSize; i++ ){
        cv::rectangle( histImage, cv::Point( bin_w*i, hist_h), cv::Point( bin_w*i + bin_w, hist_h - cvRound(hist.at<float>(i)) ), cv::Scalar(255), -1);
    }

    double max_entropy = - std::log2(1.0/256.0);
    //std::cout << 100*calculateEntropy(hist)/8.0 << std::endl;
    if (calculateEntropy(hist) >= 0.6*max_entropy){
        //Balls with stripes
        if(whitePercentage >= 5 && whitePercentage <= 15){
            return 1;
        }
        //White ball
        else if (whitePercentage > 15) {
            return 2;
        } 
        //balls with solid colors + black one
        else {
            return 3;
        }  
    } else {
        return -1;
    }
}

std::vector<cv::Vec3f> detectBalls(cv::Mat& img, cv::Mat& output, int segmentation){

    cv::Mat src = img.clone();
    
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(src, src, cv::Size(11,11), 1);
    cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,11,2);

    std::vector<cv::Vec3f> circles;
    HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 10, 150, 10, 7, 15);

    std::vector<cv::Vec3f> detected_balls;
    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>> white_balls;
    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>> solid_balls;

    for(int i = 0; i < circles.size(); i++){
        //c[0] = x coor, c[1] = y coor, c[2] = radius
        cv::Vec3f c = circles[i];
        cv::Rect ball(c[0]-c[2], c[1]-c[2], c[2], c[2]);
        //crop the image
        cv::Mat roi = src(ball);
        cv::Scalar mean;
        cv::Scalar stddev;
        cv::meanStdDev(roi, mean, stddev);

        if(0 < mean[0] && mean[0] < 160){
            //std::cout<<"Pass"<<mean<<std::endl;
            cv::Point center = cv::Point(c[0], c[1]);
            //circle certer
            //circle(output, center, 1, cv::Scalar(0,100,100));
            //circle radius
            float radius = c[2]; 
            cv::Point2i top_left(c[0]-radius, c[1]-radius);
            //cv::Point2i bottom_right(c[0]+radius, c[1]+radius);
            cv::Point2i bottom_right(c[0]+10, c[1]+10);

            cv::Rect ball_square(c[0]-radius, c[1]-radius,bottom_right.x - top_left.x, bottom_right.y - top_left.y);
            // cv::Mat ball_detected = output(ball_square);
            // cv::imshow("Ball", ball_detected);
            // cv::imshow("Histogram", histogram_cal(output(ball_square)));
            // cv::waitKey(0);

            //int class_ball = histogram_cal(output(ball_square));
            int class_ball = histogram_cal(img(ball_square));
            // std::cout << class_ball << std::endl;
            if (class_ball != -1){
                if (class_ball == 1)
                    if(segmentation)
                        cv::circle(output, center, 10, cv::Scalar(0,0,255),cv::FILLED);
                    else
                        cv::rectangle(output, top_left, bottom_right, cv::Scalar(0,0,255), 2);
                else if (class_ball == 2)
                    white_balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0));
                else if (class_ball == 3)
                    solid_balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0));
                //Save detected balls
                detected_balls.push_back(c);
            }
        }   
    }

    //detect the single white ball
    if(white_balls.size()>0)
        detectWhiteBall(white_balls, segmentation, img, output);
    
    //detect the black ball
    if(solid_balls.size()>0)
        detectBlackBall(solid_balls, segmentation, img, output);

    return detected_balls;
}



void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>>& white_balls, int& segmentation, cv::Mat& img, cv::Mat& output){
    cv::Mat gray;
    cv::Mat thresh;
    //detect the single white ball
    if(white_balls.size()>0){
        for (auto& ball_info : white_balls) {
            cv::cvtColor(img(std::get<0>(ball_info)), gray, cv::COLOR_BGR2GRAY);
            //cv::imshow("ball", thresh);
            //cv::waitKey(0);

            cv::threshold(gray, thresh, 240, 255, cv::THRESH_BINARY);
            std::get<3>(ball_info) = cv::countNonZero(thresh);
        }

        // Sort the vector based on the fourth element
        std::sort(white_balls.begin(), white_balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>& b) {
            return std::get<3>(a) > std::get<3>(b);
        });

        //plot the bounding box of the white ball
        if(segmentation){
            cv::circle(output, cv::Point2i(std::get<2>(white_balls[0]).x-5, std::get<2>(white_balls[0]).y-5), 10, cv::Scalar(255,255,255),cv::FILLED);
            //for the others, plot the segmented balls with stripes
            for (size_t i = 1; i < white_balls.size(); ++i)
                cv::circle(output, cv::Point2i(std::get<2>(white_balls[i]).x-5, std::get<2>(white_balls[i]).y-5), 10, cv::Scalar(0,0,255),cv::FILLED);
        } 
        else{
            cv::rectangle(output, std::get<1>(white_balls[0]), std::get<2>(white_balls[0]), cv::Scalar(255,255,255), 2);
            //for the others, plot the bounding boxes of the balls with stripes
            for (size_t i = 1; i < white_balls.size(); ++i)
                cv::rectangle(output, std::get<1>(white_balls[i]), std::get<2>(white_balls[i]), cv::Scalar(0,0,255), 2);
        }
        //cv::waitKey(0);
    }
}


void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>>& solid_balls, int& segmentation, cv::Mat& img, cv::Mat& output){
    cv::Mat gray;
    cv::Mat thresh;
    
    for (auto& ball_info : solid_balls) {
            // cv::cvtColor(output(std::get<0>(ball_info)), gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img(std::get<0>(ball_info)), gray, cv::COLOR_BGR2GRAY);
            //cv::imshow("ball", thresh);
            //cv::waitKey(0);

            cv::threshold(gray, thresh, 20, 255, cv::THRESH_BINARY_INV);
            std::get<3>(ball_info) = cv::countNonZero(thresh);
        }

        // Sort the vector based on the fourth element
        std::sort(solid_balls.begin(), solid_balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>& b) {
            return std::get<3>(a) > std::get<3>(b);
        });

        std::cout<<"New image"<<std::endl;
        for(auto& ball : solid_balls)
            std::cout<<std::get<3>(ball)<<std::endl;

        if(segmentation){
            cv::circle(output, cv::Point2i(std::get<2>(solid_balls[0]).x-5, std::get<2>(solid_balls[0]).y-5), 10, cv::Scalar(0,0,0),cv::FILLED);
            //for the others, plot the segmented balls with stripes
            for (size_t i = 1; i < solid_balls.size(); ++i)
                cv::circle(output, cv::Point2i(std::get<2>(solid_balls[i]).x-5, std::get<2>(solid_balls[i]).y-5), 10, cv::Scalar(255,127,0),cv::FILLED);
        } 
        else{
            cv::rectangle(output, std::get<1>(solid_balls[0]), std::get<2>(solid_balls[0]), cv::Scalar(0,0,0), 2);
            //for the others, plot the bounding boxes of the balls with stripes
            for (size_t i = 1; i < solid_balls.size(); ++i)
                cv::rectangle(output, std::get<1>(solid_balls[i]), std::get<2>(solid_balls[i]), cv::Scalar(255,127,0), 2);
        }
}