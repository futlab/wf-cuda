#include <vector>
using namespace std;
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../cuda_utils.h"


TEST(MatBufferTest, writeRead)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost, cmp;
    for (int x = 0; x < 20; x++)
        cv::line(srcHost, cv::Point(x * 100 % 640, x * 70 % 480), cv::Point(x * 350 % 640, x * 700 % 480), cv::Scalar(x * 72 % 255, x * 7 % 255, x * 214 % 255));
    MatBuffer mb(srcHost.size(), srcHost.type());
    mb.write(srcHost);
    mb.read(dstHost);
    cv::compare(srcHost, dstHost, cmp, cv::CMP_NE);
    cv::cvtColor(cmp, cmp, cv::COLOR_RGB2GRAY);
    EXPECT_EQ(int(0), cv::countNonZero(cmp));
}

/*TEST(OpenCVCudaTest, execMad)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost;

}*/

/*TEST(OpenCVCudaTest, blur)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost;
    for (int x = 0; x < 20; x++)
        cv::line(srcHost, cv::Point(x * 100 % 640, x * 70 % 480), cv::Point(x * 350 % 640, x * 700 % 480), cv::Scalar(x * 72 % 255, x * 7 % 255, x * 214 % 255));
    //cv::UMat dst, src;
    cv::gpu::GpuMat dst, src;
    src.upload(srcHost);
    //srcHost.copyTo(src);
    cv::gpu::blur(src, dst, cv::Size(3, 3));
    cv::blur(srcHost, dstHost, cv::Size(3, 3));
    cv::Mat cmp, dstCopy;
    dst.download(dstCopy);
    cv::compare(dstCopy, dstHost, cmp, cv::CMP_NE);
    cv::cvtColor(cmp, cmp, cv::COLOR_RGB2GRAY);
    EXPECT_EQ(0, cv::countNonZero(cmp));
}*/
