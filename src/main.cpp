#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>

#include "detect_table.h"

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

    Mat first_frame;
    cap.read(first_frame);

    Mat table(first_frame.rows, first_frame.cols, CV_8UC3);
    detectTable(first_frame, first_frame);
    imshow("First frame", first_frame);
    waitKey(0);
 
    // while(1){
 
    //     Mat frame;
    //     cap >> frame;
    //     if (frame.empty()){
    //         break;
    //     }
    
    //     imshow( "Frame", frame );
    
    //     char c = (char)waitKey(25); 
    //     if(c==27){ // esc to exit video
    //         break;
    //     }
    // }
    
    // cap.release();
    // destroyAllWindows();

    return 0;
}
