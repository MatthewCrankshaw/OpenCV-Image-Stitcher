#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "feature_matcher.h"

enum Mode {FEATURE, STITCH};

Mode mode = STITCH;
int lastImage;
int currentImage;

std::vector<cv::Mat> leftImgs, rightImgs, matchedImgs;
std::vector<cv::Mat> imagesLeft, imagesRight;
std::vector<cv::Mat> originalLeft, originalRight;

void handleChangeTrackbar(int, void *);
void featureMatchMode(std::vector<cv::String> filenamesLeft, std::vector<cv::String> filenamesRight);
void stitchImagesMode(std::vector<cv::String> filenamesLeft, std::vector<cv::String> filenamesRight);
void createWindow(std::string name, int width = 600, int height = 600);
void loadImages(std::vector<cv::Mat> &images, std::vector<cv::String> filenames);
void createTrackbar(int current, int last);

int main(int argc, char ** argv)
{
    std::string path1 = "res/left/*", path2 = "res/right/*";
    std::vector<cv::String> filenamesRight, filenamesLeft;

    cv::glob(path1, filenamesLeft, true);
    cv::glob(path2, filenamesRight, true);

    if(filenamesLeft.size() == 0 || filenamesRight.size() == 0){
        std::cout << "Files not read" << std::endl;
    }else{
        std::cout << "Number of left side images: " << filenamesLeft.size() << std::endl;
        std::cout << "Number of right side images: " << filenamesRight.size() << std::endl;
    }

    currentImage = 0;
    if(mode == FEATURE) {
        featureMatchMode(filenamesLeft, filenamesRight);
    } else if(mode == STITCH) {
        stitchImagesMode(filenamesLeft, filenamesRight);
    }
    return 0;
}

void featureMatchMode(std::vector<cv::String> filenamesLeft, std::vector<cv::String> filenamesRight) {
    createWindow("matchedImg");

    loadImages(imagesLeft, filenamesLeft);
    loadImages(imagesRight, filenamesRight);

    if(filenamesLeft.size() != filenamesRight.size()){
        std::cout << "not the same number of images in left and right" << std::endl;
        exit(1);
    }

    lastImage =  filenamesLeft.size()-1;

    feature_matcher matcher = feature_matcher(1080, 1920);

    for(size_t i = 0; i < filenamesLeft.size(); i++){
        cv::Mat matched;
        matcher.getMatchesSIFT(imagesLeft[i], imagesRight[i], 1000, matched);
        //matcher.getMatchesORB(img1[i], img2[i], 1000, matched);
        //matcher.getMatchesSURF(img1[i], img2[i], 10000, matched);
        matchedImgs.push_back(matched);
        std::cout << "Detecting features for: " << i << std::endl;
    }

    matcher.writeDataFile();


    createTrackbar(currentImage, lastImage);

    cv::waitKey(0);
}

void stitchImagesMode(std::vector<cv::String> filenamesLeft, std::vector<cv::String> filenamesRight) {
    //setup windows
    createWindow("window");
    createWindow("img1");
    createWindow("img2");

    loadImages(originalLeft, filenamesLeft);
    loadImages(originalRight, filenamesRight);

    //check that there is the same number of images on the left and right side
    if(filenamesLeft.size() != filenamesRight.size()){
        std::cout << "not the same number of images in left and right" << std::endl;
        exit(1);
    }

    //slider max should be the maximum number  of images
    lastImage = filenamesLeft.size()-1;

    //Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(5000, 3, 0.04, 10, 1.6);
    //Ptr<Feature2D> f2d = xfeatures2d::SURF::create(1500, 4, 3, false, false);
    cv::Ptr<cv::Feature2D> f2d = cv::ORB::create(10000);

    //loop though all images
    for(size_t i = 0; i < filenamesLeft.size(); i++){

        //make sure to push back an empty mat otherwise will segment fault
        imagesLeft.push_back(cv::Mat());
        imagesRight.push_back(cv::Mat());
        //resize the image for efficiency
        resize(originalLeft[i], imagesLeft[i], cv::Size(1080, 1920), 0, 0, cv::INTER_NEAREST);
        resize(originalRight[i], imagesRight[i], cv::Size(1080, 1920), 0, 0, cv::INTER_NEAREST);

        //offset that will be used in the resulting images to place it in the center
        int offsetx = 100;
        int offsety = 100;

        //resulting images resize
        cv::Mat reszImg = (cv::Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
        warpAffine(imagesLeft[i], imagesLeft[i], reszImg, cv::Size(2*imagesLeft[i].cols, 1.2*imagesLeft[i].rows));

        std::cout << "INFO: Detected Keypoints for " << i << std::endl;
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        f2d->detect(imagesLeft[i], keypoints1);
        f2d->detect(imagesRight[i], keypoints2);

        std::cout << "INFO: Computed descriptors for " << i << std::endl;
        cv::Mat descriptors1, descriptors2;
        f2d->compute(imagesLeft[i], keypoints1, descriptors1);
        f2d->compute(imagesRight[i], keypoints2, descriptors2);

        std::cout << "INFO: Matched descriptors for " << i << std::endl;
        cv::BFMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        cv::Mat index;
        int nbMatch = int(matches.size());
        cv::Mat tab(nbMatch, 1, CV_32F);
        for(int j = 0; j < nbMatch; j++){
            tab.at<float>(j, 0) = matches[j].distance;
        }
        sortIdx(tab, index, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
        std::vector<cv::DMatch> bestMatches;

        if(matches.size() < 100){
            std::cout << "Error: not enough matches for " << i << " Number of matches = " << matches.size() << std::endl;
            exit(1);
        }
        std::cout << "Number of matches = " << matches.size() << std::endl;

        for(int j = 0; j < 100; j++){
            bestMatches.push_back(matches[index.at<int>(j,0)]);
        }
        std::cout << "INFO: Computed best matches for " << i << std::endl;

        std::vector<cv::Point2f> dst_pts;
        std::vector<cv::Point2f> source_pts;
        for(auto it = bestMatches.begin(); it != bestMatches.end(); it++){
            dst_pts.push_back(keypoints1[it->queryIdx].pt);
            source_pts.push_back(keypoints2[it->trainIdx].pt);
        }

        cv::Mat H = findHomography(source_pts, dst_pts, cv::RANSAC);
        std::cout << "INFO: Found Homography for " << i << " : " << H << std::endl;

        cv::Mat wim2;
        warpPerspective(imagesRight[i], wim2, H, imagesLeft[i].size());

        //Nieve aproach at stitching together
        for(int x = 0; x < imagesLeft[i].cols; x++){
            for(int y = 0; y < imagesLeft[i].rows; y++){
                cv::Vec3b colorimg1 = imagesLeft[i].at<cv::Vec3b>(cv::Point(x, y));
                cv::Vec3b colorimg2 = wim2.at<cv::Vec3b>(cv::Point(x, y));
                if(norm(colorimg1)==0){
                    imagesLeft[i].at<cv::Vec3b>(cv::Point(x,y)) = colorimg2;
                }
            }
        }

    }

    char trackbarName[50];
    sprintf(trackbarName, "Image: %d", lastImage);

    cv::createTrackbar(trackbarName, "window", &currentImage, lastImage, handleChangeTrackbar);
    handleChangeTrackbar(currentImage, 0);

    cv::imshow("img1", originalLeft[1]);
    cv::imshow("img2", originalRight[1]);
    cv::imshow("window", imagesLeft[1]);
    cv::waitKey(0);
}

void handleChangeTrackbar(int, void *) {
    if (mode == FEATURE) {
        imshow("matchedImg", matchedImgs[currentImage]);
    } else if (mode == STITCH) {
        cv::imshow("window", imagesLeft[currentImage]);
        cv::imshow("img1", originalLeft[currentImage]);
        cv::imshow("img2", originalRight[currentImage]);
    }
}

void createWindow(std::string name, int width, int height) {
    cv::namedWindow(name, 0);
    cv::resizeWindow(name, cv::Size(width, height));
}

void loadImages(std::vector<cv::Mat> &images, std::vector<cv::String> filenames) {
    for(size_t i = 0; i < filenames.size(); i++){
        cv::Mat image = cv::imread(filenames[i], 1);
        images.push_back(image);
        std::cout << "Loaded Image (" << filenames[i] << ")" << std::endl;
    }
}

void createTrackbar(int current, int last) {
    char trackbarName[50];
    sprintf(trackbarName, "Image: %d", lastImage);

    cv::createTrackbar(trackbarName, "imgLeft", &current, lastImage, handleChangeTrackbar);
    handleChangeTrackbar(current, 0);
}
