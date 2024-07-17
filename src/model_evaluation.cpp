#include "model_evaluation.hpp"

double getIoU(cv::Vec4f ground_truth, cv::Vec4f predicted){

    cv::Point2f ls_ground_truth = cv::Point2f(ground_truth[0], ground_truth[1]);
    cv::Point2f rb_ground_truth = cv::Point2f(ground_truth[0] + ground_truth[2], ground_truth[1] + ground_truth[3]);

    cv::Point2f ls_predicted = cv::Point2f(predicted[0], predicted[1]);
    cv::Point2f rb_predicted = cv::Point2f(predicted[0] + predicted[2], predicted[1] + predicted[3]);

    int ix1 = std::max(static_cast<int>(ls_ground_truth.x), static_cast<int>(ls_predicted.x));
    int iy1 = std::max(static_cast<int>(ls_ground_truth.y), static_cast<int>(ls_predicted.y));
    int ix2 = std::min(static_cast<int>(rb_ground_truth.x), static_cast<int>(rb_predicted.x));
    int iy2 = std::min(static_cast<int>(rb_ground_truth.y), static_cast<int>(rb_predicted.y));

    int i_height = std::max(static_cast<int>(iy2 - iy1), 0);
    int i_width = std::max(static_cast<int>(ix2 - ix1), 0);

    double area_of_intersection = i_height * i_width;

    double gt_height = rb_ground_truth.y - ls_ground_truth.y;
    double gt_width = rb_ground_truth.x - ls_ground_truth.x;

    double pred_height = rb_predicted.y - ls_predicted.y;
    double pred_width = rb_predicted.x - ls_predicted.x;

    double area_of_union = gt_height * gt_width + pred_height * pred_width - area_of_intersection;

    if (area_of_union <= 0) {
        return 0.0;  
    }
     
    double iou = area_of_intersection / area_of_union;
     
    return iou;

}

void evaluate_one_frame(std::string path_true, std::string path_predicted, std::vector<int>& evaluations, int current_class){

    std::vector<cv::Vec4f> true_balls;
    std::vector<cv::Vec4f> predicted_balls;
    std::vector<int> true_classes;
    std::vector<int> predicted_classes;

    std::ifstream file_true(path_true);
    std::ifstream file_predicted(path_predicted);

    std::string line;

    while (getline(file_true, line)) {
        std::istringstream iss(line);
        int x, y, w, h, id;
        iss >> x >> y >> w >> h >> id;
        true_balls.push_back(cv::Vec4i(x, y, w, h));
        true_classes.push_back(id);
    }

    while (getline(file_predicted, line)) {
        std::istringstream iss(line);
        int x, y, w, h, id;
        iss >> x >> y >> w >> h >> id;
        predicted_balls.push_back(cv::Vec4i(x, y, w, h));
        predicted_classes.push_back(id);
    }

    // true_positive = 1
    // false_positive = 2
    // false_negative = 3

    for (size_t i = 0; i < predicted_balls.size(); i++) {

        if (current_class == predicted_classes[i]) {
            
            double best_iou = 0.0;
            int ground_truth_class = -1;

            for (size_t j = 0; j < true_balls.size(); j++) {

                double current_iou = getIoU(true_balls[j], predicted_balls[i]);

                if (current_iou > best_iou) {
                    best_iou = current_iou;
                    ground_truth_class = true_classes[j];
                }

            }

            // std::cout << best_iou << std::endl;
            if (best_iou >= 0.5) {
                // std::cout << ground_truth_class << " " << current_class << std::endl;
                if (ground_truth_class == current_class) {
                    evaluations.push_back(1);
                } else {
                    evaluations.push_back(2);
                }
            
            } else {
                evaluations.push_back(2);
            }

        }

    }

    for (size_t i = 0; i < true_balls.size(); i++) {

        if (current_class == true_classes[i]) {

            double best_iou = 0.0;

            for (size_t j = 0; j < predicted_balls.size(); j++) {

                double current_iou = getIoU(true_balls[i], predicted_balls[j]);

                if (current_iou > best_iou) {
                    best_iou = current_iou;
                }

            }

            // std::cout << best_iou << std::endl;
            if (best_iou < 0.5) {
                evaluations.push_back(3);
            }

        }

    }

}

double mAP() {

    std::string filename = "../docs/paths_localization.txt";
    std::ifstream file(filename);
    std::vector<std::string> paths;

    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string path;
        iss >> path;
        paths.push_back(path);
    }

    std::vector<int> classes = {1,2,3,4};
    std::vector<double> avg_precisions;

    for (auto& current_class: classes) {

        std::vector<int> evaluations;

        for (size_t i = 0; i < paths.size(); i += 2) {
            evaluate_one_frame(paths[i], paths[i+1], evaluations, current_class);
        }
        
        int total_gt = 0; // Total ground truth
        for (size_t i = 0; i < evaluations.size(); i ++) {
            
            if (evaluations[i] == 1 || evaluations[i] == 3) {
                total_gt++;
            }
        }

        std::vector<float> precision;
        std::vector<float> recall;

        float tp = 0.0;
        float fp = 0.0;
        float fn = 0.0;    
        
        for (size_t i = 0; i < evaluations.size(); i ++) {
            // std::cout << evaluations[i] << std::endl;
            if (evaluations[i] == 1) {
                tp++;
            } else if (evaluations[i] == 2){
                fp++;
            } else {
                fn++;
            }
            
            precision.push_back(tp/(tp+fp));
            recall.push_back(tp/total_gt);
        }

        std::vector<double> rt = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        std::vector<double> pt;
        for (size_t i = 0; i < rt.size(); i++) {
            double max_precision = 0;
            for (size_t j = 0; j < recall.size(); j++){
                if (recall[j] >= rt[i] && precision[j] > max_precision){
                    max_precision = precision[j];
                }
            }
            pt.push_back(max_precision);
        }

        double average_precision = 0.0;
        for (size_t i = 0; i < pt.size(); i ++) {
            average_precision += pt[i]/11;
        }

        avg_precisions.push_back(average_precision);

    }

    std::cout << "AVERAGE PRECISION: " << avg_precisions[0] << " " << avg_precisions[1] << " " << avg_precisions[2] << " " << avg_precisions[3] << std::endl;
    double mAP_value = 0.0;
    for (auto& avg_precision: avg_precisions) {
        mAP += avg_precision/classes.size();
    }
    std::cout << mAP_value << std::endl;

    return mAP_value;
}

double iou_segmentation(std::string ground_truth_path, std::string predicted_path, int current_class){

    cv::Mat ground_truth = cv::imread(ground_truth_path);
    cv::Mat predicted = cv::imread(predicted_path);

    int tp = 0;
    int fp = 0;
    int fn = 0;

    for (size_t i = 0; i < predicted.rows; i++) {
        
        for (size_t j = 0; j < predicted.cols; j++) {

            if (predicted.at<cv::Vec3b>(i, j)[0] == current_class && predicted.at<cv::Vec3b>(i, j)[0] == current_class) {
                tp++;
            } else if (predicted.at<cv::Vec3b>(i, j)[0] == current_class && predicted.at<cv::Vec3b>(i, j)[0] != current_class){
                fp++;
            } else if (predicted.at<cv::Vec3b>(i, j)[0] != current_class && predicted.at<cv::Vec3b>(i, j)[0] == current_class) {
                fn++;
            }

        }

    }

    double iou = tp/(tp+fp+fn);
    return iou; 

}

void read_mask(std::string filename){

    std::string filename;
    std::ifstream file(filename);
    std::vector<std::string> paths;

    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string path;
        iss >> path;
        paths.push_back(path);
    }

    std::vector<int> classes = {0,1,2,3,4,5};
    // double mIoU = 0.0;

    for (auto& current_class: classes) {

        for (size_t i = 0; i < paths.size(); i += 2) {

            double iou = iou_segmentation(paths[i], paths[i+1], current_class);
        }

    }

}


void evaluate() {

    map_value = mAP();
    read_mask("../docs/paths_localization.txt");

};