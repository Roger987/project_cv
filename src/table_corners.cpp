#include "table_corners.h"

double linesDist(cv::Vec2f line1, cv::Vec2f line2){
    return (abs(line1[0] - line2[0]) + abs(line1[1] - line2[1]));
}

double pointsDist(cv::Point pt1, cv::Point pt2){
    return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}

cv::Point findIntersection(cv::Vec2f line1, cv::Vec2f line2){
    
    float ct1 = cosf(line1[1]);     
    float st1 = sinf(line1[1]);    
    float ct2 = cosf(line2[1]);     
    float st2 = sinf(line2[1]);

    float angle = fabs(line1[1] - line2[1]);
    if (angle > CV_PI) {
        angle = 2 * CV_PI - angle;
    }

    if (angle < CV_PI/12 || angle > (1.0/2.0)*CV_PI){
        return cv::Point(-1, -1);
    }

    float d = ct1*st2 - st1*ct2;

    if(d!=0.0f) {   
        int x = static_cast<int> ((st2*line1[0] - st1*line2[0]) / d);
        int y = static_cast<int> ((-ct2*line1[0] + ct1*line2[0]) / d);
        return cv::Point(x, y);
    } else { 
        return cv::Point(-1, -1);
    }

}

void sortPointsCounterClockwise(std::vector<cv::Point>& points){

    double x = 0.0;
    double y = 0.0;
    for (size_t i = 0; i < points.size(); i++){
        x += points[i].x;
        y += points[i].y;
    }

    cv::Point center = cv::Point(x/points.size(), y/points.size());

    std::sort(points.begin(), points.end(), [center](cv::Point const& p1, cv::Point const& p2)
                     { return std::atan2(p1.y - center.y, p1.x - center.x) > std::atan2(p2.y - center.y, p2.x - center.x); });
}


std::vector<std::vector<cv::Point>> tableCorners(cv::Mat& src){
    
    cv::Mat gray, color_dst;
    cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::medianBlur(gray, gray, 5);
    cv::Mat canny;
    Canny(gray, canny, 50, 200, 3);
    cvtColor(canny, color_dst, cv::COLOR_GRAY2BGR );

    std::vector<cv::Vec2f> lines; 
    cv::HoughLines(canny, lines, 1, CV_PI/180, 100, 0, 0); 

    // Too many similar lines, we need to remove some
    std::vector <cv::Vec2f> linesPruned;
    linesPruned.push_back(lines[0]);
    float delta = 50;

    for (int i = 1; i < lines.size(); i++) {
        cv::Vec2f current_line = lines[i];
        bool flag = true;
        for (int j = 0; j < linesPruned.size(); j++){
            double dist = linesDist(current_line, linesPruned[j]);
            if (dist < delta){
                flag = false;
            }
        }
        if (flag){
            linesPruned.push_back(current_line);
        }
    }

    // Find the intersections between the lines
    std::vector <cv::Point> intersections;
    for (size_t i = 0; i < linesPruned.size() - 1; i++){
        for (size_t j = i+1; j < linesPruned.size(); j++) {
            cv::Point intersec = findIntersection(linesPruned[i], linesPruned[j]);
            if (intersec.x >= 0 && intersec.y >= 0 && intersec.x <= src.cols && intersec.y <= src.rows){
                intersections.push_back(intersec);
                //std::cout << intersec.x << " " << intersec.y << std::endl;
            }
        }
    }

    // To draw the polygon, the points must be ordered clockwise
    sortPointsCounterClockwise(intersections);

    cv::Mat output = src;
    std::vector<std::vector<cv::Point>> corners;
    if (!intersections.empty()) {
        corners = {intersections};
        //cv::fillPoly(output, corners, cv::Scalar(255, 0, 0));
    }

    return corners;

}