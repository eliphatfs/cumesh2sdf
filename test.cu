#include "main.cuh"

__global__ void test_point_to_tri_kernel()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1)
    {
        point_to_tri_dist_sqr(
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 2, 0),
            make_float3(1.5f, 0.4f, 1.0f)
        );
    }
    else if (i < 2)
    {
        point_to_tri_dist_sqr(
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 2, 0),
            make_float3(0.5f, 0.4f, -0.5f)
        );
    }
}

int main()
{
    test_point_to_tri_kernel<<<512, 1>>>();
    return;
}
