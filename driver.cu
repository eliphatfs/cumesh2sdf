#include "main.cuh"
#include <fstream>
#include <iterator>
#include <vector>


int main(int argc, char ** argv)
{
    std::ifstream fi(argv[argc - 1]);
    int F;
    fi >> F;
    // copies all data into buffer
    float3 * tris;
    cudaMallocManaged(&tris, F * 3 * sizeof(float3));

    for (int i = 0; i < F; i++)
        for (int j = 0; j < 3; j++)
            fi >> tris[3 * i + j].x >> tris[3 * i + j].y >> tris[3 * i + j].z;
    cudaDeviceSynchronize();

    // for (int i = 0; i < buffer.size() / 3 / sizeof(float); i++)
    //     printf("%.2f %.2f %.2f\n", tris[i].x, tris[i].y, tris[i].z);

    float * gridDist = rasterize_tris(tris, F, 128, 4.0f / 128);
    cudaDeviceSynchronize();

    std::ofstream fo("output.txt", std::fstream::trunc);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
        {
            for (int k = 0; k < 128; k++)
                fo << gridDist[i * 128 * 128 + j * 128 + k] << '\t';
            fo << '\n';
        }
    cudaFree(tris);
    cudaFree(gridDist);
    return 0;
}
