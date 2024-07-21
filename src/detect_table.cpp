//Author: Giulio Capovilla

#include "detect_table.h"

//Usefull struct
struct Pixel {
    int x, y;
    Pixel(int x, int y) : x(x), y(y) {}
};

cv::Vec3b mostFrequentColorFun(const cv::Mat& image) { 
//Function used to find the most frequent color in the central area of the image - the table center
    std::map<cv::Vec3b, int, bool(*)(const cv::Vec3b&, const cv::Vec3b&)> color_count(
        [](const cv::Vec3b& a, const cv::Vec3b& b) -> bool {
            return a.val> b.val;
        }
    );//Create ordered color dictionary

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);//Check every pixel color and add it to the color dictionary
            bool found = false;
            for (auto& pair : color_count) {
                if (pair.first == color) {
                    pair.second++;
                    found = true;
                    break;
                }
            }

            if (!found) {
                color_count[color] = 1;
            }
        }
    }

    cv::Vec3b mostFrequentColor;
    int maxCount = 0; //Find which color is the most frequent
    for (const auto& pair : color_count) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostFrequentColor = pair.first;
        }
    }

    return mostFrequentColor;
}
    
cv::Mat process_general(cv::Mat img) { 
//Process the image with an binary otsu filter to use region growing
    cv::Mat output_Image;
    std::vector<cv::Mat> channels;
    std::vector<cv::Mat> trash_channels;
    split(img, channels);
    split(img, trash_channels); 

    for (int i = 0; i < channels.size(); ++i) {
        double otsu_thresh_val = cv::threshold(channels[i], trash_channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double bias = 10; //Add a bias to the otsu optimal threshold to get the borders of the table, that are usually slightly darker
        double new_thresh_val = otsu_thresh_val - bias;
        cv::threshold(channels[i], channels[i], new_thresh_val, 255, cv::THRESH_BINARY);
    }
    
    merge(channels, output_Image);
    return output_Image;
    }

cv::Mat regionGrowing(const cv::Mat image, cv::Vec3b color, bool start_from_center, bool color_region) {
//Apply a single region growing starting from the center of the image (where the table is present)
//Used also to calculate the inverse mask, it removes hands and  external objects from the table
    int rows = image.rows;
    int cols = image.cols;
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));//Check if a pixel has been visited yet
    std::vector<std::vector<int>> classes(rows, std::vector<int>(cols, -1));//Set the class of all pixels to -1. Region growing pixels will be set to 0

    int region_class = 0;

    int dx[9] = { 1, 0, -1, 0, 5, -5, -20, 35, -35}; //Higher values used to "jump" in same color clusters that are not togheter with the table center cluster
    int dy[9] = { 0, 1, 0, -1, 5, -5, -20, 35, -35};
    if (!start_from_center){ //Used to create the inverse_mask, it stops the juming feature
        dx[4] = 0;
        dx[5] = 0;
        dx[6] = 0;
        dx[7] = 0;
        dx[8] = 0;

        dy[4] = 0;
        dy[5] = 0;
        dy[6] = 0;
        dy[7] = 0;
        dy[8] = 0;
    }
    int first_pixel_x = 0;
    int first_pixel_y = 0;

    if (start_from_center){//set the initial pixel in the centered square
        first_pixel_x = int(image.cols/2) -40;
        first_pixel_y = int(image.rows/2) -40;
    }

    for (int i = first_pixel_x; i < image.rows; ++i) {
        if (image.at<cv::Vec3b>(first_pixel_y, i)==color){
            first_pixel_x=i;
            break;
        }
    }
    std::queue<Pixel> q;
    q.push(Pixel(first_pixel_y, first_pixel_x));
    visited[first_pixel_y][first_pixel_x] = true;
    classes[first_pixel_y][first_pixel_x] = region_class;

    while (!q.empty()) {//Region growing algorithm
        Pixel pixel = q.front();
        q.pop();

        for (int k = 0; k < 9; ++k) {
            int nx = pixel.x + dx[k];
            int ny = pixel.y + dy[k];

            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && !visited[nx][ny] && image.at<cv::Vec3b>(nx, ny) == image.at<cv::Vec3b>(first_pixel_y, first_pixel_x)) {
                q.push(Pixel(nx, ny));
                visited[nx][ny] = true;
                classes[nx][ny] = region_class;
            }
        }
    }
    cv::Mat output_image=image.clone();
    //Color all pixels that are not in the region growing of black
    for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            if (classes[j][i]==-1){
                output_image.at<cv::Vec3b>(j, i)={0,0,0};
            }
        }
    }    
    //Color of white only the pixels that are not part of the region
    if (color_region){
        for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            if (classes[j][i]!=-1){
                output_image.at<cv::Vec3b>(j, i)={255,255,255};
            }
        }
    }  
    }
    
    return output_image;
}

void detectTable(cv::Mat& src, cv::Mat& output){
    src = process_general(src); //Apply the binary filter on input channels
    int y_center = int(src.rows/2); 
    int x_center = int(src.cols/2);
    cv::Rect roi(x_center, y_center, 80, 80);
    cv::Mat cropped_image = src(roi);//Create the cropped image of the image center
    cv::Vec3b table_color = mostFrequentColorFun(cropped_image);//Find the most frequent color of the 80x80 square
    cv::Mat output_img = regionGrowing(src, table_color, true, false);//Call for region growing, it creates the mask found
    output = output_img;
}