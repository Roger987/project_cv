#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>

#include "detect_table.h"
#include "table_corners.h"
#include "detect_contours.h"

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
    Mat first_frame;
    cap.read(first_frame);
    blur(first_frame, first_frame, cv::Size(9,9));

    // Uses region growing to detect the table area
    detectTable(first_frame, first_frame);
    
    // Gets the corners of the table based on the detected table region
    vector<vector<Point>> corners = tableCorners(first_frame);

    // Gets the contours of the table
    vector<vector<Point>> contours = detectContours(first_frame.rows, first_frame.cols, corners);
 
    while(1){
 
        Mat frame;
        cap >> frame;
        if (frame.empty()){
            break;
        }
    
        fillPoly(frame, corners, cv::Scalar(49, 124, 76));
        drawContours(frame, contours, -1, Scalar(0, 255, 255), 2);
        
        imshow( "Frame", frame );
    
        char c = (char) waitKey(25); 
        if(c==27){ // esc to exit video
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();

    return 0;
}
