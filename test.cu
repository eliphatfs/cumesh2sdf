#include "main.cuh"
#include <fstream>

__global__ void test_point_to_tri_kernel()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1)
    {
        point_to_tri_dist_sqr(
            make_float3(1.1475736f, -0.32425225f, -0.43132582f),
            make_float3(1.0385184f, -0.43871883f, 0.63011706f),
            make_float3(0.5103249f, -0.43593457f, -0.58324504f),
            make_float3(0.15494743f, 0.37816253f, -0.8877857f)
        );
    }
    else if (i < 2)
    {
        point_to_tri_dist_sqr(
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 2, 0),
            make_float3(1.5f, 0.4f, 1.0f)
        );
    }
    else if (i < 3)
    {
        point_to_tri_dist_sqr(
            make_float3(0, 0, 0), make_float3(1, 0, 0), make_float3(0, 2, 0),
            make_float3(0.5f, 0.4f, -0.5f)
        );
    }
}

__global__ void test_point_to_tri_data_kernel(
    const float3 * tris, const float3 * pts, const int N,
    float * outDist
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= N) return;
    outDist[x] = sqrt(point_to_tri_dist_sqr(tris[x * 3], tris[x * 3 + 1], tris[x * 3 + 2], pts[x]));
}

void test_point_to_tri_data()
{
    std::ifstream fi("tript.txt");
    int F;
    fi >> F;
    float3 * pts;
    float3 * tris;
    float * outDist;
    cudaMallocManaged(&pts, sizeof(float3) * F);
    cudaMallocManaged(&tris, sizeof(float3) * 3 * F);
    cudaMallocManaged(&outDist, sizeof(float) * F);
    for (int i = 0; i < F; i++)
        fi >> pts[i].x >> pts[i].y >> pts[i].z;
    for (int i = 0; i < F; i++)
        for (int j = 0; j < 3; j++)
            fi >> tris[i * 3 + j].x >> tris[i * 3 + j].y >> tris[i * 3 + j].z;
    test_point_to_tri_data_kernel<<<F / 256 + 1, 256>>>(tris, pts, F, outDist);
    cudaDeviceSynchronize();
    std::ofstream fo("bug/dist.txt", std::fstream::trunc);
    for (int i = 0; i < F; i++)
        fo << outDist[i] << '\n';
    cudaFree(pts);
    cudaFree(tris);
    cudaFree(outDist);
}

int main()
{
    test_point_to_tri_kernel<<<1, 1>>>();
    test_point_to_tri_data();
    return 0;
}
