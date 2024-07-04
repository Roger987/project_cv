#include "find_perspective.h"

cv::Mat findPerspective(cv::Mat src, std::vector<std::vector<cv::Point>> corners){

    double ratio = sqrt(pow(corners[0][3].x - corners[0][2].x,2) + pow(corners[0][3].y - corners[0][2].y,2))/sqrt(pow(corners[0][0].x - corners[0][3].x, 2) + pow(corners[0][0].y - corners[0][3].y, 2));
    // std::cout << "Ratio: " << ratio << std::endl;
    // std::cout << "Rounded ratio: " << round(ratio) << std::endl;

    std::vector<cv::Point2f> new_table;
    if (round(ratio) == 1){
        new_table = {
            cv::Point2f(800, 400), // bottom-right
            cv::Point2f(800, 0), // top-right
            cv::Point2f(0, 0), // top-left
            cv::Point2f(0, 400), // bottom-left
        };
    } else {
        new_table = {
            cv::Point2f(0, 400), // bottom-left
            cv::Point2f(800, 400), // bottom-right
            cv::Point2f(800, 0), // top-right
            cv::Point2f(0, 0), // top-left
        };
    }

    std::vector<cv::Point2f> corners_flat;
    for (const auto& corner : corners[0]) {
        corners_flat.push_back(cv::Point2f(corner.x, corner.y));
    }
    
    // cv::imshow("Original", src);
    // cv::waitKey(0);

    cv::Mat M = cv::getPerspectiveTransform(corners_flat, new_table);

    // cv::Mat warp;
    // cv::warpPerspective(src, warp, M, cv::Size(800, 400));
    // circle(warp, cv::Point2f(0, 0), 10, cv::Scalar(0, 255, 0), -1);
    // cv::imshow("Warped", warp);
    // cv::waitKey(0);

    return M;
}
