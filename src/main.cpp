#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "feature_matcher.h"
#include "ImageLoader.h"

enum Mode {FEATURE, STITCH};

Mode mode = FEATURE;

cv::Mat leftImg, rightImg, matchedImg;
cv::Mat imageLeft, imageRight;
cv::Mat originalLeft, originalRight;

void featureMatchMode(cv::String filenameLeft, cv::String filenameRight, ImageLoader loader);

void stitchImagesMode(cv::String filenameLeft, cv::String filenameRight, ImageLoader loader);

void createWindow(std::string name, int width = 600, int height = 600);

void stitchImage(cv::Mat homography);

int main(int argc, char ** argv)
{
    ImageLoader loader = ImageLoader();

    std::string left = "res/left/1.jpg", right = "res/right/1.jpg";

    if(mode == FEATURE) {
        featureMatchMode(left, right, loader);
    } else if(mode == STITCH) {
        stitchImagesMode(left, right, loader);
    }
    return 0;
}

void featureMatchMode(cv::String filenameLeft, cv::String filenameRight, ImageLoader loader) {
    createWindow("matchedImg");

    imageLeft = loader.loadImage(filenameLeft);
    imageRight = loader.loadImage(filenameRight);

    feature_matcher matcher = feature_matcher(1080, 1920);

    matcher.getMatchesSIFT(imageLeft, imageRight, 1000, matchedImg);
    matcher.writeDataFile();

    cv::imshow("matchedImg", matchedImg);
    cv::waitKey(0);
}

void stitchImagesMode(cv::String filenameLeft, cv::String filenameRight, ImageLoader loader) {
    //setup windows
    createWindow("window");
    createWindow("img1");
    createWindow("img2");

    originalLeft = loader.loadImage(filenameLeft);
    originalRight = loader.loadImage(filenameRight);

    cv::Ptr<cv::Feature2D> f2d = cv::ORB::create(10000);

    resize(originalLeft, imageLeft, cv::Size(1080, 1920), 0, 0, cv::INTER_NEAREST);
    resize(originalRight, imageRight, cv::Size(1080, 1920), 0, 0, cv::INTER_NEAREST);

    //offset that will be used in the resulting images to place it in the center
    int offsetx = 100;
    int offsety = 100;

    //resulting images resize
    cv::Mat reszImg = (cv::Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(imageLeft, imageLeft, reszImg, cv::Size(2*imageLeft.cols, 1.2*imageLeft.rows));

    std::cout << "INFO: Detected Keypoints " << std::endl;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    f2d->detect(imageLeft, keypoints1);
    f2d->detect(imageRight, keypoints2);

    std::cout << "INFO: Computed descriptors "<< std::endl;
    cv::Mat descriptors1, descriptors2;
    f2d->compute(imageLeft, keypoints1, descriptors1);
    f2d->compute(imageRight, keypoints2, descriptors2);

    std::cout << "INFO: Matched descriptors " << std::endl;
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
        std::cout << "Error: not enough matches. Number of matches = " << matches.size() << std::endl;
        exit(1);
    }
    std::cout << "Number of matches = " << matches.size() << std::endl;

    for(int j = 0; j < 100; j++){
        bestMatches.push_back(matches[index.at<int>(j,0)]);
    }
    std::cout << "INFO: Computed best matches" << std::endl;

    std::vector<cv::Point2f> dst_pts;
    std::vector<cv::Point2f> source_pts;
    for(auto it = bestMatches.begin(); it != bestMatches.end(); it++){
        dst_pts.push_back(keypoints1[it->queryIdx].pt);
        source_pts.push_back(keypoints2[it->trainIdx].pt);
    }

    cv::Mat homography = findHomography(source_pts, dst_pts, cv::RANSAC);
    std::cout << "INFO: Found Homography: " << homography << std::endl;

    stitchImage(homography);

    cv::imshow("img1", originalLeft);
    cv::imshow("img2", originalRight);
    cv::imshow("window", imageLeft);
    cv::waitKey(0);
}

void createWindow(std::string name, int width, int height) {
    cv::namedWindow(name, 0);
    cv::resizeWindow(name, cv::Size(width, height));
}

void stitchImage(cv::Mat homography) {
    cv::Mat wim2;
    warpPerspective(imageRight, wim2, homography, imageLeft.size());

    //Nieve aproach at stitching together
    for(int x = 0; x < imageLeft.cols; x++){
        for(int y = 0; y < imageLeft.rows; y++){
            cv::Vec3b colorimg1 = imageLeft.at<cv::Vec3b>(cv::Point(x, y));
            cv::Vec3b colorimg2 = wim2.at<cv::Vec3b>(cv::Point(x, y));
            if(norm(colorimg1)==0){
                imageLeft.at<cv::Vec3b>(cv::Point(x,y)) = colorimg2;
            }
        }
    }
}
