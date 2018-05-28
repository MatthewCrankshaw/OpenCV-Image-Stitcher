#include "feature_matcher.h"

feature_matcher::feature_matcher()
{
    //ctor
}

feature_matcher::~feature_matcher()
{
    //dtor
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

    int minHessian = 400;
    Ptr<ORB> detector = ORB::create(minHessian);
    detector->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 0.15f;
    matches.erase(matches.begin()+goodMatches, matches.end());

    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);

}
