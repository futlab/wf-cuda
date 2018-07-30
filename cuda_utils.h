#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <opencv2/core.hpp>


class MatBuffer
{
private:
        cv::Size size_;
        int type_;
        void *devPtr_;
public:
        inline void *data() { return devPtr_; }
        inline const void *data() const { return devPtr_; }
        inline const cv::Size &size() const { return size_; }
        size_t bytesCount() const;
        inline int type() const { return type_; }
        inline bool empty() const { return !size_.width && !size_.height; }
        MatBuffer() : type_(0), devPtr_(nullptr) {}
        MatBuffer(const MatBuffer &source);
        MatBuffer(const cv::Size size, int type = CV_8U);
        MatBuffer& operator = (const MatBuffer &buf);
        //MatBuffer(MatBuffer&& buf) noexcept : Buffer(std::move(buf)), size_(buf.size_), type_(buf.type_), devPtr(devPtr) {}
        void read(cv::Mat &result);
        cv::Mat read();
        cv::Mat readScaled();
        void fill(uint value = 0);
        void write(const cv::Mat &source);
        void copyTo(MatBuffer &buf) const;
        ~MatBuffer();
};


#endif // CUDA_UTILS_H
