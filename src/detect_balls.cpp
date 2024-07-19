#include "detect_balls.hpp"

//Classification's classes:
//1 : white ball
//2 : black ball
//3 : solid balls
//4 : balls with stripes 
//5 : playing field


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
    
    cv::Mat hsv, lab;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);

    int whitePixelCount = cv::countNonZero(thresh);

    int totalPixels = img.rows * img.cols;

    // Compute the percentage of white pixels
    double whitePercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    cv::Mat hist;
    cv::calcHist( &gray, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);

    int hist_w = 256, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    cv::Mat histImage( hist_h, hist_w, CV_8UC1, cv::Scalar(0) );

    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 0; i < histSize; i++ ){
        cv::rectangle( histImage, cv::Point( bin_w*i, hist_h), cv::Point( bin_w*i + bin_w, hist_h - cvRound(hist.at<float>(i)) ), cv::Scalar(255), -1);
    }

    double max_entropy = - std::log2(1.0/256.0);
    if (calculateEntropy(hist) >= 0.6*max_entropy){
        //Balls with stripes
        if(whitePercentage >= 3 && whitePercentage <= 15){
            return 4;
        }
        //White ball
        else if (whitePercentage > 15) {
            return 1;
        } 
        //balls with solid colors + black one
        else{
            return 3;

        }
    } else {
        return -1;
    }
}

void detectBalls(cv::Mat& img, cv::Mat& output, int segmentation, std::vector<cv::Vec4f>& classified_balls) {
    cv::Mat src = img.clone();
    //cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    //cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3);
    //cv::GaussianBlur(src, src, cv::Size(3, 3), 1);
    std::vector<cv::Mat> channels;
    std::vector<cv::Mat> trash_channels;
    split(src, channels);
    split(src, trash_channels); //to have the same dimension as channels

    for (int i = 0; i < channels.size(); ++i) {
        //cv::equalizeHist(channels[i], channels[i]);
        //cv::threshold(channels[i], channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double otsu_thresh_val = cv::threshold(channels[i], trash_channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double bias = 0;  
        double new_thresh_val = otsu_thresh_val - bias;
        //cv::medianBlur(channels[i], channels[i], 9);
        cv::GaussianBlur(channels[i], channels[i], cv::Size(5, 5), 1);
        cv::adaptiveThreshold(channels[i], channels[i], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, 2);
        //cv::adaptiveThreshold(channels[i], channels[i], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3);
        //cv::threshold(channels[i], channels[i], 50, 255, cv::THRESH_BINARY);
    }
    /*    for (int i = 0; i < channels.size(); ++i) {
        //cv::equalizeHist(channels[i], channels[i]);
        //cv::threshold(channels[i], channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double otsu_thresh_val = cv::threshold(channels[i], trash_channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double bias = -10;  
        double new_thresh_val = otsu_thresh_val - bias;
        //cv::threshold(channels[i], channels[i], new_thresh_val, 255, cv::THRESH_BINARY);
        cv::adaptiveThreshold(channels[i], channels[i], 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3);
    }*/
    merge(channels, src);
    int radius = 2;  // Adjust the radius according to your circular object size
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*radius, 2*radius));
    //cv::morphologyEx(src, src, cv::MORPH_OPEN, structuringElement);
    
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    //cv::medianBlur(src, src, 5);
    //cv::threshold(src, src, 90, 255, cv::THRESH_BINARY);
    //cv::adaptiveThreshold(src, src, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 3);
    cv::GaussianBlur(src, src, cv::Size(7, 7), 1);
    //cv::erode(src, src, 45);
    //cv::dilate(src,src, 5);
    //cv::Canny(src, src, 200, 200, 3, false);
    // cv::imshow("src", src);
    // cv::waitKey(0);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(src, circles, cv::HOUGH_GRADIENT, 1, 16, 130, 20, 4, 15);

    float avg_radius = 0;
    for(size_t i = 0; i < circles.size(); i++){
        avg_radius += circles[i][2];
    }
    avg_radius = int(avg_radius/circles.size());

    //update all balls parameters
    for(size_t i = 0; i < circles.size(); i++){
        circles[i][2] = avg_radius;
    }

    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>> white_balls;
    std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>> solid_balls;

    for(const auto& c : circles){
        float radius = c[2]; 
        cv::Rect ball_bbox(c[0] - c[2], c[1] - c[2], radius * 2, radius * 2);
        cv::Mat roi = output(ball_bbox);
        cv::Scalar mean, stddev;
        cv::meanStdDev(roi, mean, stddev);

        cv::Point center(c[0], c[1]);
        cv::Point2i top_left(c[0] - radius, c[1] - radius);
        cv::Point2i bottom_right(c[0] + radius, c[1] + radius);
        cv::Rect ball_square(c[0] - radius, c[1] - radius, bottom_right.x - top_left.x, bottom_right.y - top_left.y);

        int class_ball = histogram_cal(img(ball_square));
        if(class_ball != -1){
            cv::Vec4f temp(c[0], c[1], c[2], class_ball);
            classified_balls.push_back(temp);
            size_t index = classified_balls.size() - 1;

            if(class_ball == 4){
                if(segmentation)
                    cv::circle(output, center, 10, cv::Scalar(0, 0, 255), cv::FILLED);
                else
                    cv::rectangle(output, top_left, bottom_right, cv::Scalar(0, 0, 255), 2);
            }else if(class_ball == 1)
                white_balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0, index));
            else if(class_ball == 3)
                solid_balls.push_back(std::make_tuple(ball_square, top_left, bottom_right, 0, index));
        }
           
    }

    if(!white_balls.empty())
        detectWhiteBall(white_balls, segmentation, img, output, classified_balls);
    
    if(!solid_balls.empty())
        detectBlackBall(solid_balls, segmentation, img, output, classified_balls);
}

void detectWhiteBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& white_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls) {
    cv::Mat bgr[3], threshB, threshG, threshR;
    cv::Mat roi;
    int roi_size;
    if(!white_balls.empty()){
        for (auto& ball_info : white_balls) {
            
            roi = img(std::get<0>(ball_info));
            
            cv::Mat hsv;
            cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

            cv::Mat white_mask;
            cv::Scalar lower_white = cv::Scalar(0, 0, 168);
            cv::Scalar upper_white = cv::Scalar(172, 111, 255);
            cv::inRange(hsv, lower_white, upper_white, white_mask);
            int whitePixelCount = cv::countNonZero(white_mask);
            int totalPixels = roi.rows * roi.cols;
            std::get<3>(ball_info) = whitePixelCount;
        }

        //put in first position the ball with higher amount of white pixels
        std::sort(white_balls.begin(), white_balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& b) {
            return std::get<3>(a) > std::get<3>(b);
        });

        if(segmentation){
            //true white ball
            cv::circle(output, cv::Point2i(std::get<2>(white_balls[0]).x - 5, std::get<2>(white_balls[0]).y - 5), 10, cv::Scalar(255, 255, 255), cv::FILLED);
            //Label the remaining balls as balls with stripes
            for (size_t i = 1; i < white_balls.size(); i++) {
                cv::circle(output, cv::Point2i(std::get<2>(white_balls[i]).x - 5, std::get<2>(white_balls[i]).y - 5), 10, cv::Scalar(0, 0, 255), cv::FILLED);
                classified_balls[std::get<4>(white_balls[i])][3] = 4;
            }
        }else{
            cv::rectangle(output, std::get<1>(white_balls[0]), std::get<2>(white_balls[0]), cv::Scalar(255, 255, 255), 2);
            //Label the remaining balls as balls with stripes
            for (size_t i = 1; i < white_balls.size(); i++) {
                cv::rectangle(output, std::get<1>(white_balls[i]), std::get<2>(white_balls[i]), cv::Scalar(0, 0, 255), 2);
                classified_balls[std::get<4>(white_balls[i])][3] = 4;
            }
        }
    }
}

void detectBlackBall(std::vector<std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>>& solid_balls, int segmentation, cv::Mat& img, cv::Mat& output, std::vector<cv::Vec4f>& classified_balls){
    //cv::Mat gray, thresh;

    cv::Mat bgr[3], threshB, threshG, threshR;
    cv::Mat roi;
    int roi_size;

    if(!solid_balls.empty()){
        for (auto& ball_info : solid_balls){

            cv::Mat roi = img(std::get<0>(ball_info));
            cv::Mat lab;
            cv::cvtColor(roi, lab, cv::COLOR_BGR2Lab);

            cv::Mat black_mask;
            cv::inRange(lab, cv::Scalar(0, 0, 0), cv::Scalar(50, 127, 127), black_mask);

            int blackPixelCount = cv::countNonZero(black_mask);
            std::get<3>(ball_info) = blackPixelCount;
        }

        std::sort(solid_balls.begin(), solid_balls.end(), [](const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& a, const std::tuple<cv::Rect, cv::Point2i, cv::Point2i, int, size_t>& b) {
            return std::get<3>(a) > std::get<3>(b);
        });

        if (segmentation){
            //Set the correct label for the black ball
            cv::circle(output, cv::Point2i(std::get<2>(solid_balls[0]).x - 5, std::get<2>(solid_balls[0]).y - 5), 10, cv::Scalar(0, 0, 0), cv::FILLED);
            classified_balls[std::get<4>(solid_balls[0])][3] = 2;
            for (size_t i = 1; i < solid_balls.size(); i++)
                cv::circle(output, cv::Point2i(std::get<2>(solid_balls[i]).x - 5, std::get<2>(solid_balls[i]).y - 5), 10, cv::Scalar(255, 127, 0), cv::FILLED);
        }else{
            cv::rectangle(output, std::get<1>(solid_balls[0]), std::get<2>(solid_balls[0]), cv::Scalar(0, 0, 0), 2);
            classified_balls[std::get<4>(solid_balls[0])][3] = 2;
            for (size_t i = 1; i < solid_balls.size(); i++)
                cv::rectangle(output, std::get<1>(solid_balls[i]), std::get<2>(solid_balls[i]), cv::Scalar(255, 127, 0), 2);
        }
    }
}

