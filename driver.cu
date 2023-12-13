#include "main.cuh"
#include <fstream>
#include <iterator>
#include <vector>


int main()
{
    std::ifstream fi("input.txt");
    int F;
    fi >> F;
    // copies all data into buffer
    float3 * tris;
    cudaMallocManaged(&tris, F * 9 * sizeof(float));

    for (int i = 0; i < F; i++)
        for (int j = 0; j < 3; j++)
            fi >> tris[3 * i + j].x >> tris[3 * i + j].y >> tris[3 * i + j].z;
    cudaDeviceSynchronize();

    // for (int i = 0; i < buffer.size() / 3 / sizeof(float); i++)
    //     printf("%.2f %.2f %.2f\n", tris[i].x, tris[i].y, tris[i].z);

    float * gridDist = rasterize_tris(tris, F);
    cudaDeviceSynchronize();

    std::ofstream fo("output.txt", std::fstream::trunc);
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
        {
            for (int k = 0; k < 64; k++)
                fo << gridDist[i * 64 * 64 + j * 64 + k] << '\t';
            fo << '\n';
        }
    cudaFree(tris);
    cudaFree(gridDist);
    return 0;
}
