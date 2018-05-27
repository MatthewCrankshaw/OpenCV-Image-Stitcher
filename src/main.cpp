#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char ** argv)
{

    if(argc != 3){
        cout << "Please provide 2 images as arguements" << endl;
        return 1;
    }
    cv::namedWindow("Img1", 0);
    cv::namedWindow("Img2", 0);

    cv::resizeWindow("Img1", 500, 500);
    cv::resizeWindow("Img2", 500, 500);

    cv::Mat img_1 = imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img_2 = imread( argv[2], cv::IMREAD_GRAYSCALE );
    resize(img_1, img_1, Size(480, 640), 0,0, INTER_LINEAR);
    resize(img_2,img_2, Size(480, 640), 0, 0, INTER_LINEAR);

    if(!img_1.data || !img_2.data){
        cout << "Error reading images!" << endl;
        return 1;
    }

    int minHessian = 600;

    cv::Ptr<SURF> detector = SURF::create(minHessian);
    vector<KeyPoint> keypoints_1, keypoints_2;

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    Mat img_keypoints_1;
    Mat img_keypoints_2;

    drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );


    cv::imshow("Img1", img_keypoints_1);
    cv::imshow("Img2", img_keypoints_2);

    cv::waitKey(0);
    return 0;
}
