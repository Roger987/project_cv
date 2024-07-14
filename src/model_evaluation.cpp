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

std::vector<double> precision_recall(std::string path_true, std::string path_predicted){

    std::vector<cv::Vec4f> true_balls;
    std::vector<cv::Vec4f> predicted_balls;

    std::ifstream file_true(path_true);
    std::ifstream file_predicted(path_predicted);

    std::string line;

    while (getline(file_true, line)) {
        std::istringstream iss(line);
        int x, y, w, h, id;
        iss >> x >> y >> w >> h >> id;
        true_balls.push_back(cv::Vec4i(x, y, w, h));
    }

    while (getline(file_predicted, line)) {
        std::istringstream iss(line);
        int x, y, w, h, id;
        iss >> x >> y >> w >> h >> id;
        predicted_balls.push_back(cv::Vec4i(x, y, w, h));
    }

    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;

    for (size_t i = 0; i < predicted_balls.size(); i++) {

        double best_iou = 0.0;

        for (size_t j = 0; j < true_balls.size(); j++) {

            double current_iou = getIoU(true_balls[j], predicted_balls[i]);

            if (current_iou > best_iou) {
                best_iou = current_iou;
            }

        }

        // std::cout << best_iou << std::endl;
        if (best_iou >= 0.5) {
            true_positive++;
        } else {
            false_positive++;
        }

        //cv::rectangle(src, cv::Point2f(predicted_balls[i][0], predicted_balls[i][1]), cv::Point2f(predicted_balls[i][0] + predicted_balls[i][2], predicted_balls[i][1] + predicted_balls[i][3]), cv::Scalar(0, 0, 255), 2);
 
    }

    for (size_t i = 0; i < true_balls.size(); i++) {

        double best_iou = 0.0;

        for (size_t j = 0; j < predicted_balls.size(); j++) {

            double current_iou = getIoU(true_balls[i], predicted_balls[j]);

            if (current_iou > best_iou) {
                best_iou = current_iou;
            }

        }

        // std::cout << best_iou << std::endl;
        if (best_iou < 0.5) {
            false_negative++;
        }

        //cv::rectangle(src, cv::Point2f(true_balls[i][0], true_balls[i][1]), cv::Point2f(true_balls[i][0] + true_balls[i][2], true_balls[i][1] + true_balls[i][3]), cv::Scalar(255, 0, 0), 2);

    }

    double precision = 0;
    if (true_positive + false_positive != 0) {
        precision = static_cast<double>(true_positive)/(true_positive + false_positive);
    }
    double recall = 0;
    if (true_positive + false_negative != 0) {
        recall = static_cast<double>(true_positive)/(true_positive + false_negative);
    }

    // std::cout << "True Positive: " << true_positive << std::endl;
    // std::cout << "False Positive: " << false_positive << std::endl;
    // std::cout << "False Negative: " << false_negative << std::endl;
    // std::cout << "Precision: " << precision << std::endl;
    // std::cout << "Recall: " << recall << std::endl;

    // cv::imshow("Image", src);
    // cv::waitKey(0);

    return {precision, recall};

}

void evaluate() {

    std::string filename = "../docs/paths.txt";
    std::ifstream file(filename);
    std::vector<std::string> paths;

    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        std::string path;
        iss >> path;
        paths.push_back(path);
    }

    std::vector<std::vector<double>> evaluations;

    for (size_t i = 0; i < paths.size(); i += 2) {
        evaluations.push_back(precision_recall(paths[i], paths[i+1]));
    }

    for (size_t i = 0; i < evaluations.size(); i ++) {
        std::cout << evaluations[i][0] << " " << evaluations[i][1] << std::endl;
    }

    int width = 800;
    int height = 600;
    cv::Mat plot_image = cv::Mat::zeros(height, width, CV_8UC3);

    cv::line(plot_image, cv::Point(50, 550), cv::Point(750, 550), cv::Scalar(255, 255, 255), 2);
    cv::line(plot_image, cv::Point(50, 550), cv::Point(50, 50), cv::Scalar(255, 255, 255), 2);

    cv::putText(plot_image, "Recall", cv::Point(360, 580), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(plot_image, "Precision", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);


    for (const auto& pr : evaluations) {
        int x = 50 + static_cast<int>(pr[1] * 700);
        int y = 550 - static_cast<int>(pr[0] * 500);
        cv::circle(plot_image, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Precision-Recall Plot", plot_image);
    cv::waitKey(0);

};