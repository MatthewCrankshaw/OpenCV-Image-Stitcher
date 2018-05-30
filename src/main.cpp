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

int main(int argc, char ** argv)
{

    if(argc != 3){
        cout << "Please provide 2 images as arguements" << endl;
        return 1;
    }

    Mat original_1, original_2;
    Mat matches;
    Mat h1, h2;

    cv::namedWindow("matches", 0);
    cv::namedWindow("w1", 0);
    cv::namedWindow("w2", 0);

    cv::resizeWindow("matches", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("w1", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("w2", WINDOW_SIZE, WINDOW_SIZE);
    original_1 = imread( argv[1], IMREAD_COLOR );

    original_2 = imread( argv[2], IMREAD_COLOR );
    if(!original_1.data || !original_2.data){
        cout << "Error reading images!" << endl;
        return 1;
    }


    feature_matcher matcher = feature_matcher();
    matcher.getHomographyORB(original_1, original_2, matches, h1);
    //matcher.getHomographySURF(original_2, original_1, matches, h2);

    //warpPerspective(original_1, original_1, h1, Size(original_1.cols*2, original_1.rows*2));
    //warpPerspective(original_2, original_2, h2, Size(original_2.cols*2, original_2.rows));

//    Mat pano;
//    vector<Mat> imgs;
//    imgs.push_back(original_1);
//    imgs.push_back(original_2);
//
//    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, true);
//
//    Stitcher::Status status = stitcher->stitch(imgs, pano);
//
//
//    cv::imshow("matches", pano);
//    imwrite("pano.jpg", pano);
//
    cv::imshow("matches", matches);
    cv::imwrite("ORB_matches.jpg", matches);
    //cv::imshow("w1", original_1);
    //cv::imshow("w2", original_2);

    cv::waitKey(0);
    return 0;
}
