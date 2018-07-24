#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>

void execDiffint(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

#endif // FILTERS_H
