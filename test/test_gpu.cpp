#include <vector>
using namespace std;
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>


TEST(OpenCVCudaTest, blur)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::gpu::GpuMat dst, src;
    src.upload(srcHost);


}
