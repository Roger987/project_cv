#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>

#include "detect_table.h"
#include "table_corners.h"
#include "detect_contours.h"
#include "detect_balls.hpp"
#include "find_perspective.h"
#include "draw_table.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv){   

    if (argc < 2){
        std::cout << "A video filename must be provided."<<std::endl;
        return -1;
    };

    VideoCapture cap(argv[1]); 

    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // Get the first frame of the video
    Mat first_frame, src;
    cap.read(first_frame);
    first_frame.copyTo(src);
    blur(first_frame, first_frame, cv::Size(9,9));

    // Uses region growing to detect the table area
    detectTable(first_frame, first_frame);
    
    // Gets the corners of the table based on the detected table region
    vector<vector<Point>> corners = tableCorners(first_frame);

    Mat M = findPerspective(src, corners);

    // Gets the contours of the table
    vector<vector<Point>> contours = detectContours(first_frame.rows, first_frame.cols, corners);
    
    fillPoly(src, corners, cv::Scalar(49, 124, 76));
    drawContours(src, contours, -1, Scalar(0, 255, 255), 2);

    // Creates a mask to isolate the table area in order to facilitate the objects detection
    Mat mask = Mat::zeros(src.size(), CV_8UC3);
    drawContours(mask, contours, -1, cv::Scalar(255,255,255), FILLED);
    Mat cropped = Mat::zeros(src.size(), CV_8UC3);

    int segmentation = 1;
    int upvision = 1;

    std::vector<cv::Vec4f> coord_balls;
 
    while(1){
 
        Mat frame;
        cap >> frame;
        if (frame.empty()){
            break;
        }

        coord_balls.clear();

        //fillPoly(frame, corners, cv::Scalar(49, 124, 76));

        frame.copyTo(cropped, mask);

        if(segmentation == 1)
            fillPoly(frame, corners, cv::Scalar(49, 124, 76));

        //detectBalls(cropped, frame);
        // vector<Vec4f> coord_balls = detectBalls(cropped, frame, segmentation);
        detectBalls(cropped, frame, segmentation, coord_balls);
        
        drawContours(frame, contours, -1, Scalar(0, 255, 255), 2);
        //drawContours(cropped, contours, -1, Scalar(0, 255, 255), 2);

        if (upvision == 1){
           // Mat table2d = Mat(400, 800, CV_8UC3, Scalar(255, 255, 255));
            Mat table2d = drawTable(coord_balls, M);
            resize(table2d, table2d, Size(frame.cols/2, frame.rows/2), INTER_LINEAR);

            Rect roi(0, frame.rows - table2d.rows, table2d.cols, table2d.rows);
            Mat regionInterest = frame(roi);
            table2d.copyTo(regionInterest);

            //imshow("Frame", table2d);
            imshow("Frame", frame);

        } else {
            imshow("Frame", frame);
            // cv::waitKey(0);
        }

        //imshow("Frame", table2d);
        char c = (char) waitKey(25); 
        if(c==27){ // esc to exit video
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();

    return 0;
}
