#include "feature_matcher.h"
#include<opencv2/xfeatures2d.hpp>

using namespace cv::xfeatures2d;

feature_matcher::feature_matcher(int width, int height)
{
    imgWidth = width;
    imgHeight = height;
}

feature_matcher::~feature_matcher()
{
    //dtor
}

//void feature_matcher::getMatchesSURF(const Mat &src1, const Mat &src2, int param, Mat &mtchs){
//    Mat resize1, resize2;
//    Mat gray1, gray2;
//    //resize image -- only for testing purposes
//    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
//    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
//    //convert to grayscale
//    cvtColor(resize1, gray1, COLOR_BGRA2GRAY);
//    cvtColor(resize2, gray2, COLOR_BGRA2GRAY);
//
//    clock_t start;
//    start = clock();
//    Ptr<SURF> surf = SURF::create(param, 4, 3, false, false);
//    surf->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
//    surf->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);
//    float t = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
//
//    vector<vector<DMatch> > matches;
//    matchFeaturesFLANN(matches, false);
//
//    int s =0;
//    s = keypoints1.size() + keypoints2.size();
//
//    timeData += to_string(t) + "\n";
//    keypointData += to_string(s) + "\n";
//
//    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);
//}

void feature_matcher::getMatchesSIFT(const Mat &src1, const Mat &src2, int param, Mat &mtchs){
    Mat resize1, resize2;
    Mat gray1, gray2;
    //resize image -- only for testing purposes
    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
    //convert to grayscale
    cvtColor(resize1, gray1, COLOR_BGRA2GRAY);
    cvtColor(resize2, gray2, COLOR_BGRA2GRAY);

    clock_t start;
    start = clock();
    Ptr<SIFT> sift = SIFT::create(param, 3, 0.04, 10, 1.6);
    sift->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);
    float t = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);

    vector<vector<DMatch> > matches;
    matchFeaturesFLANN(matches, false);

    int s = keypoints1.size() + keypoints2.size();


    timeData += to_string(t) + "\n";
    keypointData += to_string(s) + "\n";

    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);
}

//void feature_matcher::getMatchesORB(const Mat &src1, const Mat &src2, int param, Mat &mtchs){
//    Mat resize1, resize2;
//    Mat gray1, gray2;
//
//    //resize image -- only for testing purposes
//    resize(src1, resize1, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
//    resize(src2, resize2, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
//    //convert to grayscale
//    cvtColor(resize1, gray1, COLOR_BGRA2GRAY);
//    cvtColor(resize2, gray2, COLOR_BGRA2GRAY);
//
//    clock_t start;
//    start = clock();
//    Ptr<ORB> detector = ORB::create(param, 1.2, 8, 31, 0, 2, 31, 20);
//    detector->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
//    detector->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);
//    float t = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
//
//    vector<vector<DMatch> > matches;
//    matchFeaturesFLANN(matches, true);
//
//    int s = keypoints1.size() + keypoints2.size();
//
//
//    timeData += to_string(t) + "\n";
//    keypointData += to_string(s) + "\n";
//
//    drawMatches(resize1, keypoints1, resize2, keypoints2, matches, mtchs);
//
//}

void feature_matcher::matchFeaturesFLANN(vector<vector<DMatch> > &matches, bool isORB){
    FlannBasedMatcher matcher;
    if(isORB){
        matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    }
    matcher.knnMatch(descriptors1, descriptors2, matches, 1);

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 1.0f;
    matches.erase(matches.begin()+goodMatches, matches.end());
}

void feature_matcher::matchFeaturesBruteForce(vector<DMatch> &matches){
    cv::Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

    sort(matches.begin(), matches.end());

    const int goodMatches = matches.size() * 1.00f;

//    double max_dist = 0;
//    double min_dist = 100;
//
//    for(int i = 0; i < descriptors1.rows; i++){
//        double dist = matches[i].distance;
//        if(dist<min_dist) min_dist = dist;
//        if(dist>max_dist) max_dist = dist;
//    }
//
//    cout << "Max dist is : " << max_dist << endl;
//    cout << "Min dist is : " << min_dist << endl;

    matches.erase(matches.begin()+goodMatches, matches.end());
    matcher.release();
}

void feature_matcher::writeDataFile(){
    fstream fs;
    fs.open("test.txt", fstream::out);
    fs << timeData;
    fs << keypointData;
    fs.close();
}
