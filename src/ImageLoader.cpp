#include "ImageLoader.h"

cv::Mat ImageLoader::loadImage(cv::String filename) {
    cv::Mat image = cv::imread(filename, 1);
    std::cout << "Loaded Image (" << filename << ")" << std::endl;
    return image;
}
