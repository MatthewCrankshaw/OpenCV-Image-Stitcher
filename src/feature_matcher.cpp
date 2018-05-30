#include "feature_matcher.h"
#include<opencv2/xfeatures2d.hpp>


using namespace cv::xfeatures2d;

feature_matcher::feature_matcher()
{
    imgWidth = 720;
    imgHeight = 1280;
}

feature_matcher::~feature_matcher()
{
    //dtor
}

void feature_matcher::getHomographySURF(const Mat &src1, const Mat &src2, Mat &mtchs, Mat &h){
    Mat resize1, resize2;
    Mat gray1, gray2;
    //resize image -- only for testing purposes
    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    //convert to grayscale
    cvtColor(resize1, gray1, CV_BGR2GRAY);
    cvtColor(resize2, gray2, CV_BGR2GRAY);

    Ptr<SURF> surf = SURF::create(600);
    surf->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    surf->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 1);

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 0.05f;
    matches.erase(matches.begin()+goodMatches, matches.end());

    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);
}

void feature_matcher::getHomographySIFT(const Mat &src1, const Mat &src2, Mat &mtchs, Mat &h){
    Mat resize1, resize2;
    Mat gray1, gray2;
    //resize image -- only for testing purposes
    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    //convert to grayscale
    cvtColor(resize1, gray1, CV_BGR2GRAY);
    cvtColor(resize2, gray2, CV_BGR2GRAY);

    Ptr<SIFT> sift = SIFT::create(1000);
    sift->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 1);

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 0.05f;
    matches.erase(matches.begin()+goodMatches, matches.end());

    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);
}

void feature_matcher::getHomographyORB(const Mat &src1, const Mat &src2, Mat &mtchs,Mat &h){
    Mat resize1, resize2;
    Mat gray1, gray2;
    //resize image -- only for testing purposes
    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    //convert to grayscale
    cvtColor(resize1, gray1, CV_BGR2GRAY);
    cvtColor(resize2, gray2, CV_BGR2GRAY);

    int minHessian = 600;
    Ptr<ORB> detector = ORB::create(minHessian);
    detector->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());
//
    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 0.10f;
    matches.erase(matches.begin()+goodMatches, matches.end());

    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);

    vector<Point2f> points1, points2;
    for(size_t i = 0; i < matches.size(); i++){
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    h = findHomography(points1, points2, RANSAC);

}
