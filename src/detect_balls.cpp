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
    if (whitePercentage > 20){
        std::cout << "Number of White pixels: " << whitePixelCount << " " << whitePercentage << std::endl;
        cv::imshow("ball1", img);
        cv::waitKey(0);
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
        if(whitePercentage >= 5 && whitePercentage <= 20){
            return 1;
        }
        //White ball
        else if (whitePercentage > 20) {
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

std::vector<cv::Vec3f> detectBalls(cv::Mat& src, cv::Mat& output){
    
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(src, src, cv::Size(11,11), 1);
    cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,11,2);

    std::vector<cv::Vec3f> circles;
    HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 10, 150, 10, 7, 15);

    std::vector<cv::Vec3f> detected_balls;
    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int>> class_two_balls;

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
            circle(output, center, 1, cv::Scalar(0,100,100));
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

            int class_ball = histogram_cal(output(ball_square));
            // std::cout << class_ball << std::endl;
            if (class_ball != -1){
                if (class_ball == 1){
                    cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,255,255), 2);
                } else if (class_ball == 2) {
                    class_two_balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0));
                    //cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,0), 2);
                }  else if (class_ball == 3) {
                    cv::rectangle(output, top_left, bottom_right, cv::Scalar(0,255,0), 2);
                }
                //Save detected balls
                detected_balls.push_back(c);
            }
            // cv::rectangle(output, top_left, bottom_right, cv::Scalar(255,0,255), 2);

            // //Save detected balls
            // detected_balls.push_back(c);
        }   
    }

    // cv::Mat gray;
    // cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // cv::Mat thresh;
    // cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
    // // cv::imshow("ball1", thresh);
    // // cv::waitKey(0);

    // int whitePixelCount = cv::countNonZero(thresh);

    // for(int i=0; i<class_two_balls.size(); i++){
    //     class_two_balls[i][2];
    // }
    // Now you can use the class_two_balls vector as needed
    for (auto& ball_info : class_two_balls) {
        cv::Mat gray;
        cv::Mat thresh;
        cv::cvtColor(output(std::get<0>(ball_info)), gray, cv::COLOR_BGR2GRAY);
        cv::imshow("ball", thresh);
        cv::waitKey(0);

        cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
        std::get<3>(ball_info) = cv::countNonZero(thresh);
    }

    // Sort the vector based on the fourth element
    std::sort(class_two_balls.begin(), class_two_balls.end(), [](const auto& a, const auto& b) {
        return std::get<3>(a) < std::get<3>(b);
    });

    return detected_balls;
}