#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class feature_matcher
{
    public:
        feature_matcher();
        virtual ~feature_matcher();
        void getHomographySIFT(const Mat &src1, const Mat &src2, Mat &mtchs, Mat& h);
        void getHomographyORB(const Mat &src1, const Mat &src2, Mat &matches, Mat&h);

    protected:

    private:
        Mat descriptors1, descriptors2;
        vector<KeyPoint> keypoints1, keypoints2;

        const int imgWidth = 480, imgHeight = 640;
};

#endif // FEATURE_MATCHER_H
