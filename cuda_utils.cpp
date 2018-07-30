#include <cuda_runtime.h>
#include "cuda_utils.h"

uint cvTypeSize(int type)
{
    switch (type) {
    case CV_8U: return 1;
    case CV_16U: return 2;
    case CV_8UC3: return 3;
    case CV_8UC4: return 4;
    default:
        assert(!"Unknown cv type");
        return 0;
    }
}

size_t MatBuffer::bytesCount() const {
    return uint(size_.area()) * cvTypeSize(type_);
}

MatBuffer::MatBuffer(const cv::Size size, int type) :
    size_(size), type_(type)
{
    cudaMalloc(&devPtr_, uint(size.area()) * cvTypeSize(type));
}

void MatBuffer::read(cv::Mat &result)
{
    if (result.empty() || result.type() != type_ || result.size() != size_)
        result = cv::Mat(size_, type_);
    cudaMemcpy(result.data, devPtr_, uint(size_.area()) * cvTypeSize(type_), cudaMemcpyDeviceToHost);
}

cv::Mat MatBuffer::read()
{
    assert(devPtr_ && size_.area());
    cv::Mat result(size_, type_);
    cudaMemcpy(result.data, devPtr_, uint(size_.area()) * cvTypeSize(type_), cudaMemcpyDeviceToHost);
    return result;
}

void MatBuffer::write(const cv::Mat &source)
{
    assert(!source.empty() && source.type() == type_ && source.size() == size_);
    cudaMemcpy(devPtr_, source.data, uint(size_.area()) * cvTypeSize(type_), cudaMemcpyHostToDevice);
}

void MatBuffer::copyTo(MatBuffer &buf) const
{

}

MatBuffer::~MatBuffer()
{
    if (devPtr_)
        cudaFree(devPtr_);
}
