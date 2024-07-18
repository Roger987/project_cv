#ifndef MODEL_EVALUATION_HPP
#define MODEL_EVALUATION_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

double meanIoU(std::string path_predicted, std::string path_ground_truth);

double mAP(std::string path_predicted, std::string path_ground_truth);


#endif 