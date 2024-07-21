//Roger De Almeida Matos Junior

#include "find_perspective.hpp"

cv::Mat findPerspective(cv::Mat src, std::vector<std::vector<cv::Point>> corners){

    double ratio = sqrt(pow(corners[0][3].x - corners[0][2].x,2) + pow(corners[0][3].y - corners[0][2].y,2))/sqrt(pow(corners[0][0].x - corners[0][3].x, 2) + pow(corners[0][0].y - corners[0][3].y, 2));

    std::vector<cv::Point2f> new_table;
    if (round(ratio) == 1 || (abs(corners[0][3].y - corners[0][0].y) > abs(corners[0][3].x - corners[0][2].x))){
        new_table = {
            cv::Point2f(1400, 760), // bottom-right
            cv::Point2f(1396, 142), // top-right
            cv::Point2f(155, 141), // top-left
            cv::Point2f(158, 763), // bottom-left
        };
    } else {
        new_table = {
            cv::Point2f(158, 763), // bottom-left
            cv::Point2f(1400, 760), // bottom-right
            cv::Point2f(1396, 142), // top-right
            cv::Point2f(155, 141), // top-left
        };
    }

    std::vector<cv::Point2f> corners_flat;
    for (const auto& corner : corners[0]) {
        corners_flat.push_back(cv::Point2f(corner.x, corner.y));
    }

    cv::Mat M = cv::getPerspectiveTransform(corners_flat, new_table);
    return M;
}
