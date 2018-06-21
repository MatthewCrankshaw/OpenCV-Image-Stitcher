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

//#define MODE 1
#define MODE 2
//#define MODE 3

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
    if(MODE == 1){
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
            cout << "Loaded Image (" << filenamesLeft[i] << ")" << endl;
        }

        for(size_t i = 0; i < filenamesRight.size(); i++){
            Mat temp;
            temp = imread(filenamesRight[i], 1);
            rightImgs.push_back(temp);
            cout << "Loaded Image (" << filenamesRight[i] << ")" << endl;
        }

        if(filenamesLeft.size() != filenamesRight.size()){
            cout << "not the same number of images in left and right" << endl;
            exit(1);
        }

        slider_max =  filenamesLeft.size()-1;


       feature_matcher matcher = feature_matcher(1080, 1920);

        for(size_t i = 0; i < filenamesLeft.size(); i++){
            Mat matched, h1;
            matcher.getHomographySIFT(leftImgs[i], rightImgs[i], 1000, matched, h1);
            matchedImgs.push_back(matched);
            cout << "Detecting features for: " << i << endl;
        }
        matcher.writeDataFile();


        char trackbarName[50];
        sprintf(trackbarName, "Image: %d", slider_max);

        createTrackbar(trackbarName, "imgLeft", &slider, slider_max, on_trackbar);
        on_trackbar(slider, 0);

        //imwrite("stitch.jpg", matchedImgs[2]);

        cv::waitKey(0);
    }

    if(MODE == 2){

        Mat img1, img2, orig1, orig2;

        namedWindow("window", 0);
        namedWindow("img1", 0);
        namedWindow("img2", 0);
        resizeWindow("window", Size(600, 600));
        resizeWindow("img1", Size(600, 600));
        resizeWindow("img2", Size(600, 600));

        orig1 = imread("res/left/9.jpg", 1);
        orig2 = imread("res/right/9.jpg", 1);
        if(orig1.data == NULL || orig2.data == NULL){
            cout << "Error: could not load image" << endl;
        }
        resize(orig1, img1, Size(1080, 1920), 0, 0, INTER_NEAREST);
        resize(orig2, img2, Size(1080, 1920), 0, 0, INTER_NEAREST);

        int offsetx = 1000;
        int offsety = 1000;

        Mat reszImg = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
        warpAffine(img1, img1, reszImg, Size(3*img1.cols, 2*img1.rows));

        cout << "INFO: Detecting Features" << endl;
        //Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(100, 3, 0.04, 10, 1.6);
        //Ptr<Feature2D> f2d = xfeatures2d::SURF::create(1000, 4, 3, false, false);
        Ptr<Feature2D> f2d = ORB::create(2000, 1.2, 8, 31, 0, 2, 31, 20);

        vector<KeyPoint> keypoints1, keypoints2;
        f2d->detect(img1, keypoints1);
        f2d->detect(img2, keypoints2);

        cout << "INFO: Detected Keypoints" << endl;

        Mat descriptors1, descriptors2;
        f2d->compute(img1, keypoints1, descriptors1);
        f2d->compute(img2, keypoints2, descriptors2);

        cout << "INFO: Computed descriptors" << endl;

        BFMatcher matcher;
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        cout << "INFO: Matched descriptors" << endl;

        Mat index;
        int nbMatch = int(matches.size());
        Mat tab(nbMatch, 1, CV_32F);
        for(int i = 0; i < nbMatch; i++){
            tab.at<float>(i, 0) = matches[i].distance;
        }
        sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
        vector<DMatch> bestMatches;

        for(int i = 0; i < 100; i++){
            bestMatches.push_back(matches[index.at<int>(i,0)]);
        }

        cout << "INFO: Computed 200 best matches" << endl;

        vector<Point2f> dst_pts;
        vector<Point2f> source_pts;

        for(auto it = bestMatches.begin(); it != bestMatches.end(); it++){
            dst_pts.push_back(keypoints1[it->queryIdx].pt);
            source_pts.push_back(keypoints2[it->trainIdx].pt);
        }

        Mat H = findHomography(source_pts, dst_pts, CV_RANSAC);

        cout << "INFO: Found Homography: " << H << endl;

        Mat wim2;
        warpPerspective(img2, wim2, H, img1.size());

        for(int i = 0; i < img1.cols; i++){
            for(int j = 0; j < img1.rows; j++){
                Vec3b colorimg1 = img1.at<Vec3b>(Point(i, j));
                Vec3b colorimg2 = wim2.at<Vec3b>(Point(i, j));
                if(norm(colorimg1)==0){
                    img1.at<Vec3b>(Point(i,j)) = colorimg2;
                }
            }
        }

        imshow("window", img1);
        imshow("img1", orig1);
        imshow("img2", orig2);
        waitKey(0);
    }
    return 0;
}
