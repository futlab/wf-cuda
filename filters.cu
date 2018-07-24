#include "filters.h"
#include <vector>
using namespace std;
#include <opencv2/gpu/gpu.hpp>

template <typename SRC, typename DST>
__global__ void diffint(const SRC *ghv, DST * result, uint width, uint height)
{
    ghv += mul24(width * 2, threadIdx.x); //  get_global_id(0)
    result += mul24(width, (uint)threadIdx.x) + (uint)8;

    short buffer[16], bufferv[16];
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
            sv += abs(bufferv[7 - d]) + abs(bufferv[8 + d]);
            if (abs(a) > 250 && abs(b) > 250)
                count++;
            else if (count)
                break;
        }
        int ad = abs(sa - sb), m = 2 * max(abs(sa), abs(sb));
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
    diffint<uchar, uchar> <<<src.rows / 32, 32>>>((const uchar *)src.data, (uchar *)dst.data, src.cols, src.rows);
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
