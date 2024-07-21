//Roger De Almeida Matos Junior

#include "table_corners.hpp"


double linesDist(cv::Vec2f line1, cv::Vec2f line2){
    return (abs(line1[0] - line2[0]) + abs(line1[1] - line2[1]));
}


double pointsDist(cv::Point pt1, cv::Point pt2){
    return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}


double pointLineDist(cv::Point pt, cv::Vec2f line){

    double a = cos(line[1]);
    double b = sin(line[1]);
    double c = -line[0];

    return abs(a*pt.x + b*pt.y + c) / std::sqrt(a*a + b*b);

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
    cv::Mat copy_src = src.clone();

    cv::medianBlur(gray, gray, 5);
    cv::Mat canny;
    Canny(gray, canny, 50, 200, 3);
    cvtColor(canny, color_dst, cv::COLOR_GRAY2BGR );

    std::vector<cv::Vec2f> lines; 
    cv::HoughLines(canny, lines, 1.2, CV_PI/180, 100, 0, 0); 

    // Too many similar lines, we need to remove some
    std::vector <cv::Vec2f> linesPruned;
    linesPruned.push_back(lines[0]);
    float delta = 50;
    cv::Point center(src.cols/2, src.rows/2);
    for (int i = 1; i < lines.size(); i++) {
        cv::Vec2f current_line = lines[i];
        bool flag = true;
        for (int j = 0; j < linesPruned.size(); j++){
            double dist = linesDist(current_line, linesPruned[j]);
            if (dist < delta){
                double dist1 = pointLineDist(center, current_line);
                double dist2 = pointLineDist(center, linesPruned[j]);
                if (dist1 > dist2) {
                    linesPruned[j] = current_line;
                }
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
            }
        }
    }

    std::vector <cv::Point> corner_list;
    corner_list.push_back(intersections[0]);
    if (intersections.size() > 4) {

        while (corner_list.size() < 4) {

            double max_dist = -1;
            cv::Point corner_candidate;

            for (size_t i = 0; i < intersections.size(); i++){

                double min_dist = 100000000.0;

                for (size_t j = 0; j < corner_list.size(); j++) {
                    double dist = pointsDist(intersections[i], corner_list[j]);
                    if (dist < min_dist){
                        min_dist = dist;
                    }
                }

                if (min_dist > max_dist) {
                    max_dist = min_dist;
                    corner_candidate = intersections[i];
                }

            }

            corner_list.push_back(corner_candidate);

        }

        // To draw the polygon, the points must be ordered clockwise
        sortPointsCounterClockwise(corner_list);
        return {corner_list};


    } else {
        // To draw the polygon, the points must be ordered clockwise
        sortPointsCounterClockwise(intersections);
        return {intersections};
    }
}