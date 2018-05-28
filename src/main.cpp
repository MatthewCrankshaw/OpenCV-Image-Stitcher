#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

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
    Mat img_1, img_2;

    cv::namedWindow("Img1", 0);
    cv::namedWindow("Img2", 0);
    cv::namedWindow("matches", 0);

    cv::resizeWindow("Img1", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("Img2", WINDOW_SIZE, WINDOW_SIZE);
    cv::resizeWindow("matches", WINDOW_SIZE, WINDOW_SIZE);

    original_1 = imread( argv[1], IMREAD_COLOR );
    original_2 = imread( argv[2], IMREAD_COLOR );
    if(!original_1.data || !original_2.data){
        cout << "Error reading images!" << endl;
        return 1;
    }
    resize(original_1, original_1, Size(720, 1280), 0,0, INTER_LINEAR);
    resize(original_2, original_2, Size(720, 1280), 0, 0, INTER_LINEAR);

    cvtColor(original_1, img_1, CV_BGR2GRAY);
    cvtColor(original_2, img_2, CV_BGR2GRAY);

    int minHessian = 200;
    cv::Ptr<ORB> detector = ORB::create(minHessian);
    vector<KeyPoint> keypoints_1, keypoints_2;

    Mat descriptors1, descriptors2;
    detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors1);
    detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors2);

    Mat img_keypoints_1;
    Mat img_keypoints_2;

    drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

    Mat matchesImage;
    drawMatches(original_1, keypoints_1, original_2, keypoints_2, matches, matchesImage);

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 0.15f;
    matches.erase(matches.begin()+goodMatches, matches.end());

    imwrite("ORB_matcher.jpg", matchesImage);

    cv::imshow("Img1", img_keypoints_1);
    cv::imshow("Img2", img_keypoints_2);
    cv::imshow("matches", matchesImage);

    cv::waitKey(0);
    return 0;
}
