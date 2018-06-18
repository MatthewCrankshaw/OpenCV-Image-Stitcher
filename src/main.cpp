#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "feature_matcher.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define WINDOW_SIZE 600

int slider_max;
int slider;

vector<Mat> leftImgs, rightImgs, matchedImgs;

void on_trackbar(int, void *){
    imshow("imgLeft", leftImgs[slider]);
    imshow("imgRight", rightImgs[slider]);
    imshow("matchedImg", matchedImgs[slider]);
}

int main(int argc, char ** argv)
{

    vector<String> filenamesRight, filenamesLeft;
    slider = 0;

    cv::namedWindow("imgLeft", 0);
    cv::namedWindow("imgRight", 0);
    cv::namedWindow("matchedImg", 0);

    cv::resizeWindow("imgLeft", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("imgRight", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("matchedImg", WINDOW_SIZE, WINDOW_SIZE);

    string path1 = "res/left/*", path2 = "res/right/*";
    glob(path1, filenamesLeft, true);
    glob(path2, filenamesRight, true);

    if(filenamesLeft.size() == 0 || filenamesRight.size() == 0){
        cout << "Files not read" << endl;
    }else{
        cout << "Number of left side images: " << filenamesLeft.size() << endl;
        cout << "Number of right side images: " << filenamesRight.size() << endl;
    }

    for(size_t i = 0; i < filenamesLeft.size(); i++){
        Mat temp;
        temp = imread(filenamesLeft[i], 1);
        leftImgs.push_back(temp);
    }

    for(size_t i = 0; i < filenamesRight.size(); i++){
        Mat temp;
        temp = imread(filenamesRight[i], 1);
        rightImgs.push_back(temp);
    }

    if(filenamesLeft.size() != filenamesRight.size()){
        cout << "not the same number of images in left and right" << endl;
        exit(1);
    }

    slider_max =  filenamesLeft.size()-1;


   feature_matcher matcher = feature_matcher(1080, 1920);

    for(size_t i = 0; i < filenamesLeft.size(); i++){
        Mat matched, h1;
        matcher.getHomographyORB(leftImgs[i], rightImgs[i], 1000, matched, h1);
        matchedImgs.push_back(matched);
    }
    matcher.writeDataFile();


    char trackbarName[50];
    sprintf(trackbarName, "Image: %d", slider_max);

    createTrackbar(trackbarName, "imgLeft", &slider, slider_max, on_trackbar);
    on_trackbar(slider, 0);

    cv::waitKey(0);
    return 0;
}
