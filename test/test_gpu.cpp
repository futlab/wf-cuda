#include <vector>
using namespace std;
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>


TEST(OpenCVCudaTest, blur)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost;
    cv::UMat dst, src;
    srcHost.copyTo(src);
    cv::blur(src, dst, cv::Size(3, 3));
    cv::blur(srcHost, dstHost, cv::Size(3, 3));
    cv::Mat cmp;
    cv::compare(dst, dstHost, cmp, cv::CMP_NE);
    cv::cvtColor(cmp, cmp, cv::COLOR_RGB2GRAY);
    EXPECT_EQ(0, cv::countNonZero(cmp));
}
