#include "generate_coords.hpp"
#include <fstream>
#include <iostream>

void generateCoords(std::vector<cv::Vec4f> coord_balls, std::string filename) {

    filename = filename + ".txt";

    std::ofstream myfile(filename);

    if (!myfile.is_open()) {
        std::cout << "Failed to open file" << filename << std::endl;
        return;
    }

    std::sort(coord_balls.begin(), coord_balls.end(), [](const cv::Vec4f& a, const cv::Vec4f& b) {return a[3] < b[3];});

    for (auto& ball: coord_balls){
        myfile << round(ball[0]) - round(ball[2]) << " " << round(ball[1]) - round(ball[2]) << " " << round(2*round(ball[2])) << " " << round(2*round(ball[2])) << " " << ball[3] << std::endl;
    }

    //myfile << test << std::endl;

    myfile.close();

    return;
}