#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <sstream>
#include <string>

#include "detect_table.h"
#include "table_corners.h"
#include "detect_contours.h"
#include "detect_balls.hpp"
#include "find_perspective.h"
#include "draw_table.hpp"
#include "generate_coords.hpp"
// #include "model_evaluation.hpp"

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

    stringstream ss(argv[1]); 
    string token; 
    vector<string> tokens;
    while (std::getline(ss, token, '/')) { 
        tokens.push_back(token); 
    }

    // Flags for the output
    int segmentation = 0;
    int upvision = 1;

    // Get the first frame of the video
    Mat first_frame, src, original;
    cap.read(first_frame);
    first_frame.copyTo(src);
    first_frame.copyTo(original);
    blur(first_frame, first_frame, cv::Size(9,9));

    // Uses region growing to detect the table area
    detectTable(first_frame, first_frame);
    
    // Gets the corners of the table based on the detected table region
    vector<vector<Point>> corners = tableCorners(first_frame);

    // Find the matrix to the geometric transformation
    Mat M = findPerspective(src, corners);

    // Gets the contours of the table
    vector<vector<Point>> contours = detectContours(first_frame.rows, first_frame.cols, corners);
    
    // fillPoly(src, corners, cv::Scalar(49, 124, 76));
    drawContours(src, contours, -1, Scalar(0, 255, 255), 2);
    // imshow("SRC", src);
    // waitKey(0);

    // Creates a mask to isolate the table area in order to facilitate the objects (balls) detection
    Mat mask = Mat::zeros(src.size(), CV_8UC3);
    drawContours(mask, contours, -1, cv::Scalar(255,255,255), FILLED);
    Mat cropped = Mat::zeros(src.size(), CV_8UC3);

    std::vector<cv::Vec4f> coord_balls;
    
    // Detect balls of first frame to initialize multitracking
    original.copyTo(cropped,mask);
    std::vector<cv::Ptr<cv::Tracker>> multitracker;
    detectBalls(cropped, original, segmentation, coord_balls);
    std::vector<cv::Rect> tracked_balls_bbx;
    for (size_t i = 0; i < coord_balls.size(); i++){
        int x=coord_balls[i][0];
        int y=coord_balls[i][1];
        int r=coord_balls[i][2];
        tracked_balls_bbx.push_back(cv::Rect(x-1.25*r,y-1.25*r,r*2.5,r*2.5)); 
        cv::rectangle(original, cv::Rect(x-1.25*r,y-1.25*r,r*2.5,r*2.5), cv::Scalar(0, 0, 0), 2);
    }
    for (size_t i = 0; i < tracked_balls_bbx.size(); i++){
        cv::Ptr<cv::Tracker> balltracker = TrackerCSRT::create();
        balltracker->init(original, tracked_balls_bbx[i]);
        multitracker.push_back(balltracker);
    }

    std::vector<std::vector<cv::Point2f>> trajectories(coord_balls.size());

    // int count_frame = 1;

    // Reset to the first frame
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    VideoWriter output("../Dataset/" + tokens[2] + "/output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, Size(src.cols,src.rows));
 
    while(true){
 
        Mat frame;
        cap >> frame;
        Mat segmented_frame = frame.clone();
        if (frame.empty()){
            break;
        }

    //     if (count_frame == 1) {
    //         generateCoords(coord_balls, "../Dataset/" + tokens[2] + "/bounding_boxes/frame_first");
    //     } else if (count_frame == total_frame - 1){
    //         generateCoords(coord_balls, "../Dataset/" + tokens[2] + "/bounding_boxes/frame_last");
    //     }
 
        // for (size_t i = 0; i < multitracker.size(); i++){
        //     bool updated_bbx = multitracker[i]->update(frame, tracked_balls_bbx[i]);
        //     if (updated_bbx){
        //         if (coord_balls[i][3] == 1) {
        //             cv::rectangle(frame, tracked_balls_bbx[i], cv::Scalar(255, 255, 255), 2);
        //         } else if (coord_balls[i][3] == 2) {
        //             cv::rectangle(frame, tracked_balls_bbx[i], cv::Scalar(0, 0, 0), 2);
        //         } else if (coord_balls[i][3] == 3) {
        //             cv::rectangle(frame, tracked_balls_bbx[i], cv::Scalar(255, 127, 0), 2);
        //         } else if (coord_balls[i][3] == 4) {
        //             cv::rectangle(frame, tracked_balls_bbx[i], cv::Scalar(0, 0, 255), 2);
        //         }
        //     }
        // }

        if (upvision == 1){
            
            for (size_t i = 0; i < multitracker.size(); i++){
                multitracker[i]->update(frame, tracked_balls_bbx[i]);
            }

            // Mat table2d = Mat(400, 800, CV_8UC3, Scalar(255, 255, 255));
            Mat table2d = drawTable(tracked_balls_bbx, coord_balls, trajectories, M);
            resize(table2d, table2d, Size(0.4*frame.cols, 0.4*frame.rows), INTER_AREA);

            Rect roi(0, frame.rows - table2d.rows, table2d.cols, table2d.rows);
            Mat regionInterest = frame(roi);
            table2d.copyTo(regionInterest);

        } else {

            if(segmentation == 1)
                fillPoly(segmented_frame, corners, cv::Scalar(49, 124, 76));

            drawContours(frame, contours, -1, Scalar(0, 255, 255), 2);

            for (size_t i = 0; i < multitracker.size(); i++){
                bool updated_bbx = multitracker[i]->update(frame, tracked_balls_bbx[i]);
                if (updated_bbx){
                    Scalar color;
                    if (coord_balls[i][3] == 1) {
                        color = Scalar(255,255,255);
                    } else if (coord_balls[i][3] == 2) {
                        color = Scalar(0,0,0);
                    } else if (coord_balls[i][3] == 3) {
                        color = Scalar(255,0,0);
                    } else if (coord_balls[i][3] == 4) {
                        color = Scalar(0,0,255);
                    }

                    if (segmentation) {
                        int centerX = tracked_balls_bbx[i].x + tracked_balls_bbx[i].width / 2;
                        int centerY = tracked_balls_bbx[i].y + tracked_balls_bbx[i].height / 2;
                        cv::circle(segmented_frame, Point(centerX, centerY), 10, color, cv::FILLED);
                    } else {
                        cv::rectangle(frame, tracked_balls_bbx[i], color, 2);
                    }
                }
            }

        }

        if (segmentation) {
            output.write(segmented_frame);
        } else {
            output.write(frame);
        }
    }
    
    cap.release();
    output.release();
    destroyAllWindows();

    // MODEL EVALUATION

    cout << tokens[2] << endl;
    // evaluate();
    // evaluate("../Dataset/" + tokens[2] + "/bounding_boxes/frame_first_bbox.txt", "../Dataset/" + tokens[2] + "/bounding_boxes/frame_first.txt", tokens[2], 1);
    // cout << "\n" << endl;
    // evaluate("../Dataset/" + tokens[2] + "/bounding_boxes/frame_last_bbox.txt", "../Dataset/" + tokens[2] + "/bounding_boxes/frame_last.txt", tokens[2], 0);
    // cout << "\n\n" << endl;
    return 0;
}
