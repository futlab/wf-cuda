#include <iostream>
//#include "gpu.hpp"

/*__global__ void
filter(unsigned int *input, unsigned int *od, int w, int h, int r)
{

}

extern "C"
double boxFilterRGBA(unsigned int *d_src, unsigned int *d_temp, unsigned int *d_dest, int width, int height,
                     int radius, int iterations, int nthreads, StopWatchInterface *timer)
{
    checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_array));

    // var for kernel computation timing
    double dKernelTime;

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer_kernel
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        // use texture for horizontal pass
        d_boxfilter_rgba_x<<< height / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
        d_boxfilter_rgba_y<<< width / nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);

        // sync host and stop computation timer_kernel
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpyToArray(d_tempArray, 0, 0, d_dest, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTextureToArray(rgbaTex, d_tempArray));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}*/
