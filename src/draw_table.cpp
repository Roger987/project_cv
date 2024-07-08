#include "draw_table.hpp"

cv::Mat drawTable(std::vector<cv::Vec4f> coord_balls, cv::Mat M) {

    cv::Mat table2d = cv::imread("../docs/table.png");

    std::vector<cv::Point2f> input_balls;
    std::vector<cv::Point2f> transf_coord_balls;
    
    for (size_t i = 0; i < coord_balls.size(); i++){
        input_balls.push_back(cv::Point2f(coord_balls[i][0],coord_balls[i][1]));
    }

    perspectiveTransform(input_balls, transf_coord_balls, M);

    for (size_t i = 0; i < transf_coord_balls.size(); i++){
        if (coord_balls[i][3] == 1){
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(255, 255, 255), -1);
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(0, 0, 0), 1);
        } else if (coord_balls[i][3] == 2){
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(0, 0, 0), -1);
        } else if (coord_balls[i][3] == 3){
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(250, 230, 200), -1);
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(0, 0, 0), 1);
        } else if (coord_balls[i][3] == 4){
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(150, 150, 250), -1);
            circle(table2d, transf_coord_balls[i], 16, cv::Scalar(0, 0, 0), 1);
        }
    }     

    return table2d;  
}