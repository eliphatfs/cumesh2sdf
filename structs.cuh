#pragma once
#include <cuda.h>

struct RasterizeResult
{
    float * gridDist;
    float3 * gridPseudoNormal;
    int * gridRepPoint;

    void free()
    {
        cudaFree(gridDist);
        cudaFree(gridPseudoNormal);
        cudaFree(gridRepPoint);
    }
};
