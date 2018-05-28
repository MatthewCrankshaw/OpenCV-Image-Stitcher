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
    Mat h;

    cv::namedWindow("matches", 0);

    cv::resizeWindow("matches", WINDOW_SIZE, WINDOW_SIZE);
    original_1 = imread( argv[1], IMREAD_COLOR );

    original_2 = imread( argv[2], IMREAD_COLOR );
    if(!original_1.data || !original_2.data){
        cout << "Error reading images!" << endl;
        return 1;
    }


    feature_matcher matcher = feature_matcher();
    matcher.getHomographyORB(original_1, original_2, matches, h);

    cv::imshow("matches", matches);
    cv::imwrite("ORB_goodmatches.jpg", matches);

    cv::waitKey(0);
    return 0;
}
