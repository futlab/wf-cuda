#include "helper_math.h"
#include <stdio.h>
//#include <helper_functions.h>

texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray *d_array, *d_tempArray;

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
           ((unsigned int)(rgba.z * 255.0f) << 16) |
           ((unsigned int)(rgba.y * 255.0f) <<  8) |
           ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

// row pass using texture lookups
__global__ void
boxfilterRgbaX(unsigned int *dst, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    // as long as address is always less than height, we do work
    if (y < h)
    {
        float4 t = make_float4(0.0f);

        for (int x = -r; x <= r; x++)
        {
            t += tex2D(rgbaTex, x, y);
        }

        dst[y * w] = rgbaFloatToInt(t * scale);

        for (int x = 1; x < w; x++)
        {
            t += tex2D(rgbaTex, x + r, y);
            t -= tex2D(rgbaTex, x - r - 1, y);
            dst[y * w + x] = rgbaFloatToInt(t * scale);
        }
    }
}

// column pass using coalesced global memory reads
__global__ void
boxfilterRgbaY(unsigned int *id, unsigned int *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    id = &id[x];
    od = &od[x];

    float scale = 1.0f / (float)((r << 1) + 1);

    float4 t;
    // do left edge
    t = rgbaIntToFloat(id[0]) * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += rgbaIntToFloat(id[y*w]);
    }

    od[0] = rgbaFloatToInt(t * scale);

    for (int y = 1; y < (r + 1); y++)
    {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[0]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += rgbaIntToFloat(id[(h - 1) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
}

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        printf(/*stderr,*/ "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

double boxFilterRGBA(const unsigned int *d_src, unsigned int *d_temp, unsigned int *d_dest, int width, int height,
                     int radius, int iterations, int nthreads)
{
    checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array));

    // var for kernel computation timing
    double dKernelTime;

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer_kernel
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        //sdkResetTimer(&timer);

        // use texture for horizontal pass
        boxfilterRgbaX<<< height / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
        boxfilterRgbaY<<< width / nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);

        // sync host and stop computation timer_kernel
        checkCudaErrors(cudaDeviceSynchronize());
        //dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpyToArray(d_tempArray, 0, 0, d_dest, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_tempArray));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}
