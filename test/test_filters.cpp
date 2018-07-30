#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include "../filters.h"
#ifdef TEST_WITH_GUI
#include <opencv2/highgui.hpp>
#endif

TEST(FiltersTest, mad)
{
    cv::Mat srcHost(cv::Size(1280, 720), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost, dstHostRef, cmp;
    for (int x = 0; x < 20; x++)
        cv::line(srcHost, cv::Point(x * 100 % 640, x * 70 % 480), cv::Point(x * 350 % 640, x * 700 % 480), cv::Scalar(x * 72 % 100, x * 7 % 100, x * 214 % 100));

    MatBuffer src(srcHost.size(), srcHost.type()), dst(srcHost.size(), srcHost.type());
    src.write(srcHost);
    execMad(src, dst, 2, 10);
    dstHost = dst.read();
    srcHost.convertTo(dstHostRef, CV_8UC3, 2, 10);
    /*cv::compare(dstHost, dstHostRef, cmp, cv::CMP_NE);
    cv::cvtColor(cmp, cmp, cv::COLOR_RGB2GRAY);
    EXPECT_EQ(int(0), cv::countNonZero(cmp));*/

    bool equal = cv::sum(dstHost != dstHostRef) == cv::Scalar(0,0,0,0);
    EXPECT_TRUE(equal);

#ifdef TEST_WITH_GUI
    if (!equal) {
        cv::imshow("dstRef", dstHostRef);
        cv::imshow("dst", dstHost);
        cv::waitKey();
    }
#endif
}

TEST(FiltersTest, blur)
{
    cv::Mat srcHost(cv::Size(1280, 720), CV_8UC4, cv::Scalar(0, 0, 0)), dstHost, dstHostRef, cmp;
    for (int x = 0; x < 20; x++)
        cv::line(srcHost, cv::Point(x * 100 % 640, x * 70 % 480), cv::Point(x * 350 % 640, x * 700 % 480), cv::Scalar(x * 72 % 100, x * 7 % 100, x * 214 % 100));

    MatBuffer src(srcHost.size(), srcHost.type()), dst(srcHost.size(), srcHost.type());

    src.write(srcHost);
    BlurFilter bf(cv::Size(1280, 720));
    bf.boxFilter(src, dst, 1, 1);
    dstHost = dst.read();
    cv::blur(srcHost, dstHostRef, cv::Size(3, 3));


    /*cv::compare(dstHost, dstHostRef, cmp, cv::CMP_NE);
    cv::cvtColor(cmp, cmp, cv::COLOR_RGB2GRAY);
    EXPECT_EQ(int(0), cv::countNonZero(cmp));*/

    bool equal = cv::sum(dstHost != dstHostRef) == cv::Scalar(0,0,0,0);
    EXPECT_TRUE(equal);

#ifdef TEST_WITH_GUI
    if (!equal) {
        cv::imshow("dstRef", dstHostRef);
        cv::imshow("dst", dstHost);
        cv::waitKey();
    }
#endif
}

TEST(FiltersTest, diffint)
{
    cv::Mat srcHost(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 0)), dstHost;
    for (int x = 0; x < 20; x++)
        cv::line(srcHost, cv::Point(x * 100 % 640, x * 70 % 480), cv::Point(x * 350 % 640, x * 700 % 480), cv::Scalar(x * 72 % 255, x * 7 % 255, x * 214 % 255));
    cv::UMat dst, src;
    srcHost.copyTo(src);

    /*cv::cuda::GpuMat srcGpu;
    src.copyTo(srcGpu);
    execDiffint(src, dst);*/

}
