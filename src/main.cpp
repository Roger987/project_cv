//Roger De Almeida Matos Junior
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
#include "generate_mask.hpp"
#include "model_evaluation.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv){   

    if (argc < 5){
        std::cout << "A video filename must be provided." << std::endl;
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
    int segmentation = std::stoi(argv[2]);
    int upvision = std::stoi(argv[3]);
    int evaluate = std::stoi(argv[4]);

    // Get the first frame of the video
    Mat first_frame, src, original;
    cap.read(first_frame);
    first_frame.copyTo(src);
    first_frame.copyTo(original);
    blur(first_frame, first_frame, cv::Size(9,9));

    // Uses region growing to detect the table area
    detectTable(first_frame, first_frame);
    Mat croppedoriginal = Mat::zeros(src.size(), CV_8UC3);
    original.copyTo(croppedoriginal,first_frame);
    Mat inverse_mask = regionGrowing(first_frame, Vec3b(0,0,0), false, true);
    cv::threshold(inverse_mask, inverse_mask, 200, 255, cv::THRESH_BINARY_INV);

    // Gets the corners of the table based on the detected table region
    vector<vector<Point>> corners = tableCorners(croppedoriginal);

    // Find the matrix to the geometric transformation
    Mat M = findPerspective(src, corners);

    // Gets the contours of the table
    vector<vector<Point>> contours = detectContours(first_frame.rows, first_frame.cols, corners);

    // fillPoly(src, corners, cv::Scalar(49, 124, 76));
    drawContours(src, contours, -1, Scalar(0, 255, 255), 2);
    // imshow("table", src);
    // waitKey(0);

    // Creates a mask to isolate the table area in order to facilitate the objects (balls) detection
    Mat mask = Mat::zeros(src.size(), CV_8UC3);
    drawContours(mask, contours, -1, cv::Scalar(255,255,255), FILLED);
    Mat cropped = Mat::zeros(src.size(), CV_8UC3);
    std::vector<cv::Vec4f> coord_balls;

    // Detect balls of first frame to initialize multitracking
    original.copyTo(cropped,inverse_mask);
    std::vector<cv::Ptr<cv::Tracker>> multitracker;
    detectAndClassifyBalls(cropped, original, segmentation, coord_balls);
    
    //Calculate average radius
    float avg_radius = 0;
    for(size_t i = 0; i < coord_balls.size(); i++){
        avg_radius += coord_balls[i][2];
    }
    avg_radius = int(avg_radius/coord_balls.size());

    //update all balls parameters
    for(size_t i = 0; i < coord_balls.size(); i++){
        coord_balls[i][2] = avg_radius;
    }

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

    // Trajectories vector
    std::vector<std::vector<cv::Point2f>> trajectories(coord_balls.size());

    int count_frame = 1;
    int video_length = int(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Reset to the first frame
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    // Output video
    VideoWriter output("../Dataset/" + tokens[2] + "/output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, Size(src.cols,src.rows));
 
    while(true){
 
        Mat frame;
        cap >> frame;
        Mat segmented_frame = frame.clone();
        
        if (frame.empty()){
            break;
        }

        Mat cropped_frame = Mat::zeros(frame.size(), CV_8UC3);
        frame.copyTo(cropped_frame,mask);

        if (upvision == 1){
            
            for (size_t i = 0; i < multitracker.size(); i++){
                multitracker[i]->update(cropped_frame, tracked_balls_bbx[i]);
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
                bool updated_bbx = multitracker[i]->update(cropped_frame, tracked_balls_bbx[i]);
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
                        cv::circle(segmented_frame, Point(centerX, centerY), avg_radius, color, cv::FILLED);
                    } else {
                        cv::rectangle(frame, tracked_balls_bbx[i], color, 2);
                    }
                }
            }

        }

        if (segmentation && !upvision) {
            output.write(segmented_frame);
        } else {
            output.write(frame);
        }

        if (evaluate) {
            if (count_frame == 1) {
                cout << tokens[2] << endl;
                generateCoords(coord_balls, "../Dataset/" + tokens[2] + "/bounding_boxes/frame_first.txt");
                generateMask(coord_balls, tracked_balls_bbx, avg_radius, corners, frame.cols, frame.rows, "../Dataset/" + tokens[2] + "/masks/frame_first_computed.png");
                double miou = meanIoU("../Dataset/" + tokens[2] + "/masks/frame_first_computed.png", "../Dataset/" + tokens[2] + "/masks/frame_first.png");
                double map = mAP("../Dataset/" + tokens[2] + "/bounding_boxes/frame_first.txt","../Dataset/" + tokens[2] + "/bounding_boxes/frame_first_bbox.txt");
                cout << "First frame evaluation:\n Mean IoU: " << miou << "\n mAP: " << map << endl;
            } else if (count_frame == video_length - 1){
                generateCoords(coord_balls, "../Dataset/" + tokens[2] + "/bounding_boxes/frame_last.txt");
                generateMask(coord_balls, tracked_balls_bbx, avg_radius, corners, frame.cols, frame.rows, "../Dataset/" + tokens[2] + "/masks/frame_last_computed.png");
                double miou = meanIoU("../Dataset/" + tokens[2] + "/masks/frame_last_computed.png", "../Dataset/" + tokens[2] + "/masks/frame_last.png");
                double map = mAP("../Dataset/" + tokens[2] + "/bounding_boxes/frame_last.txt","../Dataset/" + tokens[2] + "/bounding_boxes/frame_last_bbox.txt");
                cout << "Last frame evaluation:\n Mean IoU: " << miou << "\n mAP: " << map << endl;
            }
        }

        count_frame++;
    }
    
    cap.release();
    output.release();
    destroyAllWindows();

    return 0;
}
