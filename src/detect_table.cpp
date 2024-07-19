//#include "functions.h"
#include "detect_table.h"

struct Pixel {
    int x, y;
    Pixel(int x, int y) : x(x), y(y) {}
};

int colorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) {
    return std::sqrt(
        std::pow(color1[0] - color2[0], 2) +
        std::pow(color1[1] - color2[1], 2) +
        std::pow(color1[2] - color2[2], 2)
    );
}

cv::Vec3b mostFrequentColorWithThreshold(const cv::Mat& image, int threshold) {
    // Mappa per contare la frequenza dei colori
    std::map<cv::Vec3b, int, bool(*)(const cv::Vec3b&, const cv::Vec3b&)> colorCount(
        [](const cv::Vec3b& a, const cv::Vec3b& b) -> bool {
            return std::lexicographical_compare(a.val, a.val + 3, b.val, b.val + 3);
        }
    );

    // Conta i colori nell'immagine
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);

            // Verifica se un colore simile esiste già nella mappa
            bool found = false;
            for (auto& pair : colorCount) {
                if (colorDistance(pair.first, color) <= threshold) {
                    pair.second++;
                    found = true;
                    break;
                }
            }

            // Se non trovato, aggiungi il nuovo colore
            if (!found) {
                colorCount[color] = 1;
            }
        }
    }

    // Trova il colore più frequente
    cv::Vec3b mostFrequentColor;
    int maxCount = 0;
    for (const auto& pair : colorCount) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostFrequentColor = pair.first;
        }
    }

    return mostFrequentColor;
}
    
cv::Mat process_general(cv::Mat img) {  

    cv::Mat output_Image;
    std::vector<cv::Mat> channels;
    std::vector<cv::Mat> trash_channels;
    split(img, channels);
    split(img, trash_channels); //to have the same dimension as channels

    for (int i = 0; i < channels.size(); ++i) {
        //cv::equalizeHist(channels[i], channels[i]);
        //cv::threshold(channels[i], channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double otsu_thresh_val = cv::threshold(channels[i], trash_channels[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        double bias = 10;  
        double new_thresh_val = otsu_thresh_val - bias;
        cv::threshold(channels[i], channels[i], new_thresh_val, 255, cv::THRESH_BINARY);
    }
    
    merge(channels, output_Image);
    //cv::imshow("After filter", output_Image);
    //cv::waitKey(0);
    return output_Image;
    }

cv::Mat regionGrowing(const cv::Mat image, cv::Vec3b color, bool start_from_center, bool color_region) {

    int rows = image.rows;
    int cols = image.cols;
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<std::vector<int>> classes(rows, std::vector<int>(cols, -1));
    std::map<std::string, int> cluster_size;
    // togli la tupla che � inutile (in PYTHON), serve solo per fare ritornare 2 argomenti dalla funzione.
    std::tuple<std::vector<std::vector<int>>, std::map<std::string, int>> tuplee;

    int currentClass = 0;

    //per definire gli spostamenti nei vicini
    int dx[9] = { 1, 0, -1, 0, 5, -5, -10, -20,-35};
    int dy[9] = { 0, 1, 0, -1, 5, -5, -10, -20, -35};
    if (!start_from_center){
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

    if (start_from_center){
        first_pixel_x = int(image.cols/2) -40;
        first_pixel_y = int(image.rows/2) -40;
    }

    for (int i = first_pixel_x; i < image.rows; ++i) {
        if (image.at<cv::Vec3b>(first_pixel_y, i)==color){
            first_pixel_x=i;
            break;
        }
    }
    //std::cout<<first_pixel_x<<std::endl;
    // itera su tutti ipixel dell'immagine
                std::queue<Pixel> q;
    q.push(Pixel(first_pixel_y, first_pixel_x));
    visited[first_pixel_y][first_pixel_x] = true;
    classes[first_pixel_y][first_pixel_x] = currentClass;

    int count = 1; // iniziallizzo la variabile che conta quanti pixels ci sono per ogni cluster( da usare sucessivamente per selezionare i cluster pi� approrpriati

    while (!q.empty()) {
        Pixel pixel = q.front();
        q.pop();

        for (int k = 0; k < 9; ++k) {
            int nx = pixel.x + dx[k];
            int ny = pixel.y + dy[k];

            //seil vicino � all'interno dell'immagine e non � stato visitato
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && !visited[nx][ny] && image.at<cv::Vec3b>(nx, ny) == image.at<cv::Vec3b>(first_pixel_y, first_pixel_x)) {
                q.push(Pixel(nx, ny));
                visited[nx][ny] = true;
                classes[nx][ny] = currentClass;
                count++; // conta il numero di pixel che appartengono a quel cluster                            
            }
        }
    }
    //ci segniamo quando grande � questo cluster
    cluster_size[std::to_string(currentClass)] = count;
    // Avanziamo alla prossima classe        
    cv::Mat output_image=image.clone();
    for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            if (classes[j][i]==-1){
                //std::cout<<"Done untill image ";
                output_image.at<cv::Vec3b>(j, i)={0,0,0};
            }
        }
    }    

    if (color_region){
        for (int i = 0; i < image.cols; ++i) {
        for (int j = 0; j < image.rows; ++j) {
            if (classes[j][i]!=-1){
                //std::cout<<"Done untill image ";
                output_image.at<cv::Vec3b>(j, i)={255,255,255};
            }
        }
    }  
    }
    
    return output_image;
}

void detectTable(cv::Mat& src, cv::Mat& output){
    src = process_general(src);
    int y_center = int(src.rows/2); 
    int x_center = int(src.cols/2);
    cv::Rect roi(x_center, y_center, 80, 80);
    cv::Mat cropped_image = src(roi);
    cv::Vec3b table_color = mostFrequentColorWithThreshold(cropped_image, 100);
    cv::Mat output_img = regionGrowing(src, table_color, true, false);
    output = output_img;
    cv::imshow("Output", output_img);
    cv::waitKey(0);
}