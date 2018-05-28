#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/flann/flann.hpp>

using namespace std;
using namespace cv;

class feature_matcher
{
    public:
        feature_matcher();
        virtual ~feature_matcher();
        void getHomographySURF(const Mat &src1, const Mat &src2, Mat &mtchs, Mat& h);
        void getHomographySIFT(const Mat &src1, const Mat &src2, Mat &mtchs, Mat& h);
        void getHomographyORB(const Mat &src1, const Mat &src2, Mat &matches, Mat&h);

    protected:

    private:
        Mat descriptors1, descriptors2;
        vector<KeyPoint> keypoints1, keypoints2;

        int imgWidth, imgHeight;
};

#endif // FEATURE_MATCHER_H
