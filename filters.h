#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include "cuda_utils.h"

void execDiffint(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

void execMad(const MatBuffer &src, MatBuffer &dst, int a, int b);

class BlurFilter
{
private:
    unsigned int *d_temp = nullptr;
public:
    BlurFilter(const cv::Size &size);
    double boxFilter(const MatBuffer &src, MatBuffer &dst, int radius, int iterations);
};

#endif // FILTERS_H

