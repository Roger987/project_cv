//Author: Francesco Stella

#include "detect_balls.hpp"

//Classes:
//1 : white ball
//2 : black ball
//3 : solid balls
//4 : balls with stripes 
//5 : playing field

// Compute the entropy of the provided image. Used to detect false posivites balls
double calculateEntropy(const cv::Mat& histogram) {
    double entropy = 0.0;
    double total_pixels = cv::sum(histogram)[0];

    for (int i = 0; i < histogram.rows; ++i){
        double p = histogram.at<float>(i) / total_pixels;
        if (p > 0.0)
            entropy -= p * std::log2(p);
    }
    return entropy;
}

// Compute the histogram of an image and return its class
int histogramCal(cv::Mat img){
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);

    int whitePixelCount = cv::countNonZero(thresh);
    int totalPixels = img.rows * img.cols;

    // Compute the percentage of white pixels
    double whitePercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100;

    //Calculate the histogram of the image
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::Mat hist;
    cv::calcHist( &gray, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);

    int hist_w = 256, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);

    cv::Mat histImage( hist_h, hist_w, CV_8UC1, cv::Scalar(0) );

    //Normalize it to increase the contrast
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 0; i < histSize; i++ )
        cv::rectangle( histImage, cv::Point( bin_w*i, hist_h), cv::Point( bin_w*i + bin_w, hist_h - cvRound(hist.at<float>(i)) ), cv::Scalar(255), -1);

    double max_entropy = - std::log2(1.0/256.0);
    if (calculateEntropy(hist) >= 0.6*max_entropy){
        //Balls with stripes + white ball
        if(whitePercentage >= 3)
            return 4;
        //balls with solid colors + black one
        else
            return 3;
    } else 
        return -1;
}

void detectAndClassifyBalls(cv::Mat& img, cv::Mat& output, int segmentation, std::vector<cv::Vec4f>& classified_balls) {
    cv::Mat src = img.clone();
    std::vector<cv::Mat> channels;
    std::vector<cv::Mat> thresh_channels;
    split(src, channels);
    split(src, thresh_channels);

    // We use all three channels to have a more robust analysis
    for (int i = 0; i < channels.size(); ++i){
        //Apply a gaussin blur to smooth each channel
        cv::GaussianBlur(channels[i], channels[i], cv::Size(5, 5), 1);
        //Another threshold to remove remaining noise
        cv::adaptiveThreshold(channels[i], channels[i], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, 2);
    }
    
    merge(channels, src);
    // Convert the image to gray scale to be a suitable input for HoughCircles
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(src, src, cv::Size(7, 7), 1);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 16, 130, 18, 4, 15);
    
    // Vector of tuples used to identify white and black balls
    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>> balls;

    for(const auto& c : circles){
        float radius = c[2]; 
        // Extract the region of interest from the image using the coordinates of the center
        // and the radius found by HoughCircles function
        cv::Rect ball_bbox(c[0] - c[2], c[1] - c[2], radius * 2, radius * 2);
        // Retrieve upper left and bottom right corner to display the final bounding boxes or 
        // for the segmentation of the first frame
        cv::Point center(c[0], c[1]);
        cv::Point2i top_left(c[0] - radius, c[1] - radius);
        cv::Point2i bottom_right(c[0] + radius, c[1] + radius);
        cv::Rect ball_square(c[0] - radius, c[1] - radius, bottom_right.x - top_left.x, bottom_right.y - top_left.y);

        // histogramCal assign the balls to two classes: solid or stripes balls
        int class_ball = histogramCal(img(ball_square));
        if(class_ball != -1){
            cv::Vec4f temp(c[0], c[1], c[2], class_ball);
            classified_balls.push_back(temp);
            size_t index = classified_balls.size() - 1;
            balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0, index));
        }
           
    }

    // Find white ball
    if(!balls.empty())
        detectWhiteBall(balls, segmentation, img, output, classified_balls);
    
    // Find black ball
    if(!balls.empty())
        detectBlackBall(balls, segmentation, img, output, classified_balls);
}

void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls) {
    cv::Mat bgr[3], threshB, threshG, threshR;
    cv::Mat roi;
    int roi_size;
    if(!balls.empty()){
        for (auto& ball_info : balls){
            roi = img(std::get<0>(ball_info));
            // For the white ball we work in the HSV space
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
            cv::Mat white_mask;
            // Upper and lower boundaries of the white color in the HSV color space
            cv::Scalar lower_white = cv::Scalar(0, 0, 200);
            cv::Scalar upper_white = cv::Scalar(180, 100, 255);
            // The pixels inside the boundaries are set at the max value, the others to zero
            cv::inRange(hsv, lower_white, upper_white, white_mask);
            int whitePixelCount = cv::countNonZero(white_mask);
            int totalPixels = roi.rows * roi.cols;
            // Retrieve the percentage of white pixels in the roi
            std::get<3>(ball_info) = whitePixelCount*100/totalPixels;
        }

        // Put in first position the ball with higher percentage of white pixels
        std::sort(balls.begin(), balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& b){
            return std::get<3>(a) > std::get<3>(b);
        });

        if(segmentation){
            //true white ball
            classified_balls[std::get<4>(balls[0])][3] = 1;
            cv::circle(output, cv::Point2i(std::get<2>(balls[0]).x - 5, std::get<2>(balls[0]).y - 5), 10, cv::Scalar(255, 255, 255), cv::FILLED);
        }else
            classified_balls[std::get<4>(balls[0])][3] = 1;
    }
}

void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls){
    cv::Mat roi;

    if(!balls.empty()){
        for (auto& ball_info : balls){
            cv::Mat roi = img(std::get<0>(ball_info));
            cv::Mat lab;
            // For the black ball we work in the LAB color space
            cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);
            // The pixels with values outside the boundaries of in range functions are set to zero,
            // the remaining ones to the maximum value
            cv::Mat black_mask;
            cv::inRange(roi, cv::Scalar(0, 0, 0), cv::Scalar(80, 50, 100), black_mask);
            int blackPixelCount = cv::countNonZero(black_mask);
            int totalPixels = roi.rows * roi.cols;
            // Compute the percentage of white pixels that coincides with the percentage of
            // black pixels in the RGB color space
            std::get<3>(ball_info) = blackPixelCount*100/totalPixels;
        }

        std::sort(balls.begin(), balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& b){
            return std::get<3>(a) > std::get<3>(b);
        });

        if (segmentation){
            //Set the correct label for the black ball
            cv::circle(output, cv::Point2i(std::get<2>(balls[0]).x - 5, std::get<2>(balls[0]).y - 5), 10, cv::Scalar(0, 0, 0), cv::FILLED);
            classified_balls[std::get<4>(balls[0])][3] = 2;
        }else
            classified_balls[std::get<4>(balls[0])][3] = 2;
    }
}

