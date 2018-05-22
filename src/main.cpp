#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

int main(int argc, char ** argv)
{

    if(argc != 3){
        cout << "Please provide 2 images as arguements" << endl;
        return 1;
    }
    cv::namedWindow("Img1", 0);
    cv::namedWindow("Img2", 0);

    cv::resizeWindow("Img1", 500, 500);
    cv::resizeWindow("Img2", 500, 500);

    cv::Mat img_1 = imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img_2 = imread( argv[2], cv::IMREAD_GRAYSCALE );

    if(!img_1.data || !img_2.data){
        cout << "Error reading images!" << endl;
        return 1;
    }

    cv::imshow("Img1", img_1);
    cv::imshow("Img2", img_2);

    cv::waitKey(0);
    return 0;
}
