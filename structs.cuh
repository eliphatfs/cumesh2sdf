#pragma once
#include <cuda.h>

struct RasterizeResult
{
    float * gridDist;
    bool * gridCollide;

    void free()
    {
        cudaFree(gridDist);
        cudaFree(gridCollide);
    }
};
