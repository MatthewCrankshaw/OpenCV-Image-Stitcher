#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "feature_matcher.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define WINDOW_SIZE 600

//#define MODE 1
#define MODE 2

int slider_max;
int slider;

vector<Mat> leftImgs, rightImgs, matchedImgs;

std::vector<cv::Mat> img1, img2, orig1, orig2;

void on_trackbar(int, void *){
    #if(MODE == 1)
    imshow("matchedImg", matchedImgs[slider]);
    #elif(MODE == 2)
        cv::imshow("window", img1[slider]);
        cv::imshow("img1", orig1[slider]);
        cv::imshow("img2", orig2[slider]);
    #endif // MODE
}

int main(int argc, char ** argv)
{

    //mode for testing the feature matching and feature detecting algorithms
    if(MODE == 1){
        vector<String> filenamesRight, filenamesLeft;
        slider = 0;
        namedWindow("matchedImg", 0);

        resizeWindow("matchedImg", WINDOW_SIZE, WINDOW_SIZE);

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
            img1.push_back(temp);
            cout << "Loaded Image (" << filenamesLeft[i] << ")" << endl;
        }

        for(size_t i = 0; i < filenamesRight.size(); i++){
            Mat temp;
            temp = imread(filenamesRight[i], 1);
            img2.push_back(temp);
            cout << "Loaded Image (" << filenamesRight[i] << ")" << endl;
        }

        if(filenamesLeft.size() != filenamesRight.size()){
            cout << "not the same number of images in left and right" << endl;
            exit(1);
        }

        slider_max =  filenamesLeft.size()-1;


       feature_matcher matcher = feature_matcher(1080, 1920);

        for(size_t i = 0; i < filenamesLeft.size(); i++){
            Mat matched;
            matcher.getMatchesSIFT(img1[i], img2[i], 1000, matched);
            //matcher.getMatchesORB(img1[i], img2[i], 1000, matched);
            //matcher.getMatchesSURF(img1[i], img2[i], 10000, matched);
            matchedImgs.push_back(matched);
            cout << "Detecting features for: " << i << endl;
        }

        matcher.writeDataFile();


        char trackbarName[50];
        sprintf(trackbarName, "Image: %d", slider_max);

        createTrackbar(trackbarName, "imgLeft", &slider, slider_max, on_trackbar);
        on_trackbar(slider, 0);

        cv::waitKey(0);
    }

    //Mode for image stitching
    if(MODE == 2){
        //setup windows
        cv::namedWindow("window", 0);
        cv::namedWindow("img1", 0);
        cv::namedWindow("img2", 0);
        cv::resizeWindow("window", cv::Size(600, 600));
        cv::resizeWindow("img1", cv::Size(600, 600));
        cv::resizeWindow("img2", cv::Size(600, 600));

        vector<String> filenamesRight, filenamesLeft;
        slider = 0;

        vector<float> times;

        //get all of the filenames
        string path1 = "res/left/*", path2 = "res/right/*";
        glob(path1, filenamesLeft, true);
        glob(path2, filenamesRight, true);

        //check that there are images to load
        if(filenamesLeft.size() == 0 || filenamesRight.size() == 0){
            cout << "Files not read" << endl;
        }else{
            cout << "Number of left side images: " << filenamesLeft.size() << endl;
            cout << "Number of right side images: " << filenamesRight.size() << endl;
        }

        //read in the files and put them in the vector of Mat
        for(size_t i = 0; i < filenamesLeft.size(); i++){
            Mat temp;
            temp = imread(filenamesLeft[i], 1);
            orig1.push_back(temp);
            cout << "Loaded Image (" << filenamesLeft[i] << ")" << endl;
        }

        //read in the files and put them in the vector of Mat
        for(size_t i = 0; i < filenamesRight.size(); i++){
            Mat temp;
            temp = imread(filenamesRight[i], 1);
            orig2.push_back(temp);
            cout << "Loaded Image (" << filenamesRight[i] << ")" << endl;
        }

        //check that there is the same number of images on the left and right side
        if(filenamesLeft.size() != filenamesRight.size()){
            cout << "not the same number of images in left and right" << endl;
            exit(1);
        }

        //slider max should be the maximum number  of images
        slider_max =  filenamesLeft.size()-1;

        //Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(5000, 3, 0.04, 10, 1.6);
        //Ptr<Feature2D> f2d = xfeatures2d::SURF::create(1500, 4, 3, false, false);
        Ptr<Feature2D> f2d = ORB::create(10000);

        //loop though all images
        for(int i = 0; i < filenamesLeft.size(); i++){

            //make sure to push back an empty mat otherwise will segment fault
            img1.push_back(Mat());
            img2.push_back(Mat());
            //resize the image for efficiency
            resize(orig1[i], img1[i], Size(1080, 1920), 0, 0, INTER_NEAREST);
            resize(orig2[i], img2[i], Size(1080, 1920), 0, 0, INTER_NEAREST);

            //offset that will be used in the resulting images to place it in the center
            int offsetx = 100;
            int offsety = 100;

            //resulting images resize
            Mat reszImg = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
            warpAffine(img1[i], img1[i], reszImg, Size(2*img1[i].cols, 1.2*img1[i].rows));

            clock_t start;
            start = clock();

            cout << "INFO: Detected Keypoints for " << i << endl;
            vector<KeyPoint> keypoints1, keypoints2;
            f2d->detect(img1[i], keypoints1);
            f2d->detect(img2[i], keypoints2);

            cout << "INFO: Computed descriptors for " << i << endl;
            Mat descriptors1, descriptors2;
            f2d->compute(img1[i], keypoints1, descriptors1);
            f2d->compute(img2[i], keypoints2, descriptors2);

            float t = (clock() - start) / (double)(CLOCKS_PER_SEC / 100);
            cout << "Time taken " << t << endl;
            times.push_back(t);

            cout << "INFO: Matched descriptors for " << i << endl;
            BFMatcher matcher;
            vector<DMatch> matches;
            matcher.match(descriptors1, descriptors2, matches);

            Mat index;
            int nbMatch = int(matches.size());
            Mat tab(nbMatch, 1, CV_32F);
            for(int j = 0; j < nbMatch; j++){
                tab.at<float>(j, 0) = matches[j].distance;
            }
            sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
            vector<DMatch> bestMatches;

            if(matches.size() < 100){
                cout << "Error: not enough matches for " << i << " Number of matches = " << matches.size() << endl;
                exit(1);
            }
            cout << "Number of matches = " << matches.size() << endl;

            for(int j = 0; j < 100; j++){
                bestMatches.push_back(matches[index.at<int>(j,0)]);
            }
            cout << "INFO: Computed best matches for " << i << endl;

            vector<Point2f> dst_pts;
            vector<Point2f> source_pts;
            for(auto it = bestMatches.begin(); it != bestMatches.end(); it++){
                dst_pts.push_back(keypoints1[it->queryIdx].pt);
                source_pts.push_back(keypoints2[it->trainIdx].pt);
            }

            Mat H = findHomography(source_pts, dst_pts, RANSAC);
            cout << "INFO: Found Homography for " << i << " : " << H << endl;

            Mat wim2;
            warpPerspective(img2[i], wim2, H, img1[i].size());

            //Nieve aproach at stitching together
            for(int x = 0; x < img1[i].cols; x++){
                for(int y = 0; y < img1[i].rows; y++){
                    Vec3b colorimg1 = img1[i].at<Vec3b>(Point(x, y));
                    Vec3b colorimg2 = wim2.at<Vec3b>(Point(x, y));
                    if(norm(colorimg1)==0){
                        img1[i].at<Vec3b>(Point(x,y)) = colorimg2;
                    }
                }
            }

        }
        float ave = 0;
        float total = 0;
        for(auto t: times){
            total += t;
        }
        ave = total /30;
        cout << "Average time: " << ave << endl;

        char trackbarName[50];
        sprintf(trackbarName, "Image: %d", slider_max);

        createTrackbar(trackbarName, "window", &slider, slider_max, on_trackbar);
        on_trackbar(slider, 0);

        imshow("img1", orig1[1]);
        imshow("img2", orig2[1]);
        imshow("window", img1[1]);
        waitKey(0);
    }
    return 0;
}
