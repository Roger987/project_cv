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

    // Gets the contours of the table
    vector<vector<Point>> contours = detectContours(first_frame.rows, first_frame.cols, corners);

    // Creates a mask to isolate the table area in order to facilitate the objects detection
    Mat mask = Mat::zeros(src.size(), CV_8UC3);
    drawContours(mask, contours, -1, cv::Scalar(255,255,255), FILLED);
    Mat cropped = Mat::zeros(src.size(), CV_8UC3);
 
    while(1){
 
        Mat frame;
        cap >> frame;
        if (frame.empty()){
            break;
        }

        //fillPoly(frame, corners, cv::Scalar(49, 124, 76));

        frame.copyTo(cropped, mask);
        detectBalls(cropped, frame);
        
        drawContours(frame, contours, -1, Scalar(0, 255, 255), 2);
        //drawContours(cropped, contours, -1, Scalar(0, 255, 255), 2);
        
        imshow("Frame", frame);
    
        char c = (char) waitKey(25); 
        if(c==27){ // esc to exit video
            break;
        }
    }
    
    // cap.release();
    // destroyAllWindows();

    return 0;
}
