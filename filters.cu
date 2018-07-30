#include "filters.h"
//#include <vector>
//using namespace std;
//#include <opencv2/gpu/gpu.hpp>
//#include <opencv2/cudaimgproc.hpp>



// blockDim - local size
// blockIdx - group id
// threadIdx - local id

typedef unsigned char uchar;

template <typename SRC, typename DST>
__global__ void mad(const SRC *src, DST *dst, uint size, int a, int b)
{
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        dst[i] = a * src[i] + b;
}

void execMad(const MatBuffer &src, MatBuffer &dst, int a, int b)
{
    assert(src.size() == dst.size());
    mad<uchar, uchar><<<720 * 1280 / 32, 32>>>((const uchar *)src.data(), (uchar *)dst.data(), src.bytesCount(), a, b);
}

double boxFilterRGBA(const unsigned int *d_src, unsigned int *d_temp, unsigned int *d_dest, int width, int height,
                     int radius, int iterations, int nthreads);

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line);


double BlurFilter::boxFilter(const MatBuffer &src, MatBuffer &dst, int radius, int iterations)
{
    assert(src.size() == dst.size() && src.type() == CV_8UC4 && dst.type() == CV_8UC4);
    boxFilterRGBA((const unsigned int*)src.data(), d_temp, (unsigned int*)dst.data(), src.size().width, src.size().height, radius, iterations, 32);
}

BlurFilter::BlurFilter(const cv::Size &size) {
    checkCudaErrors(cudaMalloc((void **) &d_temp, (size.area() * sizeof(unsigned int))));

}



/*template <typename SRC, typename DST, int R, int C>
__device__ void blurRowPass(const SRC *src, DST *dst, int width, int height)
{
    int d = C * width * (blockIdx.x * blockDim.x + threadIdx.x);
    src += d;
    dst += d;
    DST t[C];
    for (int c = 0; c < C; c++)
        t[c] = src[c] * (R + 1);
    for (int x = 1; x < r + 1; x++) {
        for (int c = 0; c < C; c++)
            t[c] += src[c];
        src += C;
    }
    for (int x = 0; ) {
        for (int c = 0; c < C; c++)
            dst[c] = t[c];
        
        for (int c = 0; c < C; c++)
            
        
        dst += C;
        src += C;
    }
        
    

}


template <typename SRC, typename DST, int kernel>
__global__ void blur(const SRC *src, DST *dst)
{

}

void execBlur(const MatBuffer &src, MatBuffer &dst)
{

}*/

template <typename SRC, typename DST>
__global__ void diffint(const SRC *ghv, DST * result, uint width, uint height)
{
    ghv += mul24(width * 2, threadIdx.x); //  get_global_id(0)
    result += mul24(width, (uint)threadIdx.x) + (uint)8;

    SRC buffer[16], bufferv[16];
    for (int x = 0; x < 16; x++, ghv += 2) {
        buffer[x] = ghv[0];
        bufferv[x] = ghv[1];
    }
    for (int x = 16; x < width; x++, ghv += 2, result++) {

        int sa = 0, sb = 0, sv = 0, count = 0, d;
        for (d = 0; d < 8; d++) {
            int a = buffer[7 - d], b = buffer[8 + d];//, m = max(abs(a), abs(b));
            sa += a;
            sb += b;
            sv += __sad(bufferv[7 - d], 0, __sad(bufferv[8 + d], 0, 0));
            if (abs(a) > 250 && abs(b) > 250)
                count++;
            else if (count)
                break;
        }
        int ad = __sad(sa, sb, 0), m = 2 * max(abs(sa), abs(sb));
        //if (sv > abs(sa) + abs(sb)) count = 0;
        uchar r = 255 * ad / m;
        *result = (count && r > 200 && sv < m) ? d * 31 : 0;
        for (int d = 0; d < 15; d++) {
            buffer[d] = buffer[d + 1];
            bufferv[d] = bufferv[d + 1];
        }
        buffer[15] = ghv[0];
        bufferv[15] = ghv[1];
    }
}

void execDiffint(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
    typedef unsigned char uchar;
    //diffint<uchar, uchar> <<<src.rows / 32, 32>>>((const uchar *)src.data, (uchar *)dst.data, src.cols, src.rows);
}



/*__global__ void scharr5(__read_only image2d_t src, __write_only image2d_t dst)
{
        sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        const int x = get_global_id(0), y = get_global_id(1);
        __const int4 k[3] = {(int4)(-3, -6, 6, 3), (int4)(-2, -2, 2, 2), (int4)(-1, -1, 1, 1)};

        int4 rv = (int4)(0, 0, 0, 0), rh = (int4)(0, 0, 0, 0);
        for(int dx = -2; dx <= 2; ++dx) {
            int lx = x + dx;
            int4 v = convert_int4((uint4)(read_imageui(src, sampler, (int2)(lx, y - 2)).x, read_imageui(src, sampler, (int2)(lx, y - 1)).x, read_imageui(src, sampler, (int2)(lx, y + 1)).x, read_imageui(src, sampler, (int2)(lx, y + 2)).x));
            rv = mad24(v, k[abs(dx)], rv);
            if (dx)
                rh = mad24(v, (int4)(1, 2, 2, 1), rh);
            else
                rh = -rh;
        }
        int4 vh = convert_int4((uint4)(read_imageui(src, sampler, (int2)(x - 2, y)).x, read_imageui(src, sampler, (int2)(x - 1, y)).x, read_imageui(src, sampler, (int2)(x + 1, y)).x, read_imageui(src, sampler, (int2)(x + 2, y)).x));
        rh = mad24(vh, k[0], rh);

        int4 r = (int4)(rh.lo + rh.hi, rv.lo + rv.hi);
        write_imagei(dst, (int2)(x, y), (int4)(r.even + r.odd, 0, 0)); // [-5355 : 5355]
}
*/
