#include "draw_table.hpp"

double pointsDist(cv::Point2f pt1, cv::Point2f pt2){
    return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}

cv::Mat drawTable(std::vector<cv::Rect> tracked_balls_bbx, std::vector<cv::Vec4f> coord_balls, std::vector<std::vector<cv::Point2f>>& trajectories, cv::Mat M) {

    cv::Mat table2d = cv::imread("../docs/table.png");

    std::vector<cv::Point2f> input_balls;
    std::vector<cv::Point2f> transf_coord_balls;
    
    for (size_t i = 0; i < tracked_balls_bbx.size(); i++){
        int centerX = tracked_balls_bbx[i].x + tracked_balls_bbx[i].width / 2;
        int centerY = tracked_balls_bbx[i].y + tracked_balls_bbx[i].height / 2;
        input_balls.push_back(cv::Point2f(centerX,centerY));
    }

    perspectiveTransform(input_balls, transf_coord_balls, M);

    for (size_t i = 0; i < transf_coord_balls.size(); i++){
        trajectories[i].push_back(transf_coord_balls[i]);

        for (size_t j = 1; j < trajectories[i].size(); j++){
            // cv::circle(table2d, trajectories[i][j], 2, cv::Scalar(0, 0, 0), -1);
            if (j % 2 == 0) {
                cv::line(table2d, trajectories[i][j-1], trajectories[i][j], cv::Scalar(0, 0, 0), 2);
            }
        }

        if (coord_balls[i][3] == 1){
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(255, 255, 255), -1);
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(0, 0, 0), 2);
        } else if (coord_balls[i][3] == 2){
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(0, 0, 0), -1);
        } else if (coord_balls[i][3] == 3){
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(250, 230, 200), -1);
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(0, 0, 0), 2);
        } else if (coord_balls[i][3] == 4){
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(150, 150, 250), -1);
            circle(table2d, transf_coord_balls[i], 20, cv::Scalar(0, 0, 0), 2);
        }
    }     

    return table2d;  
}