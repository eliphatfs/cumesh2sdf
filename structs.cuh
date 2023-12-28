#pragma once
#include <cuda.h>

struct RasterizeResult
{
    float * gridDist;
    int * gridIdx;

    void free()
    {
        cudaFree(gridDist);
        cudaFree(gridIdx);
    }
};
