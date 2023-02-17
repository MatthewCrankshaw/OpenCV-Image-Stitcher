#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include <opencv2/opencv.hpp>

class ImageLoader
{
    public:
        cv::Mat loadImage(cv::String filename);
};

#endif // IMAGELOADER_H
