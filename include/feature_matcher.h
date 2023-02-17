#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/flann/flann.hpp>
#include<fstream>
#include<ctime>

//using namespace std;
//using namespace cv;

class feature_matcher
{
    public:
        feature_matcher(int width, int height);
        virtual ~feature_matcher();
        //void getMatchesSURF(const cv::Mat &src1, const cv::Mat &src2, int param, cv::Mat &mtchs);
        void getMatchesSIFT(const cv::Mat &src1, const cv::Mat &src2, int param, cv::Mat &mtchs);
        //void getMatchesORB(const cv::Mat &src1, const cv::Mat &src2, int param, cv::Mat &Matches);

        void writeDataFile();

    protected:

    private:
        cv::Mat descriptors1, descriptors2;
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        std::string timeData;
        std::string keypointData;

        int imgWidth, imgHeight;

        void matchFeaturesFLANN(std::vector<std::vector<cv::DMatch> > &matches, bool isORB);
        void matchFeaturesBruteForce(std::vector<cv::DMatch> &matches);
};

#endif // FEATURE_MATCHER_H
