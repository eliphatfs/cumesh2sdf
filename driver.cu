#include "main.cuh"
#include <fstream>
#include <iterator>
#include <vector>
#include <chrono>
#include <iostream>


int main(int argc, char ** argv)
{
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();
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
    const auto inputPhase = clock.now() - start;
    start = clock.now();

    // for (int i = 0; i < buffer.size() / 3 / sizeof(float); i++)
    //     printf("%.2f %.2f %.2f\n", tris[i].x, tris[i].y, tris[i].z);

    RasterizeResult rast = rasterize_tris(tris, F, 128, 4.0f / 128);
    cudaDeviceSynchronize();
    const auto rasterizePhase = clock.now() - start;
    start = clock.now();

    std::ofstream fo("output.txt", std::fstream::trunc);
    for (int i = 0; i < 128; i++)
        for (int j = 0; j < 128; j++)
        {
            for (int k = 0; k < 128; k++)
                fo << rast.gridDist[i * 128 * 128 + j * 128 + k] << '\t';
            fo << '\n';
        }
    cudaFree(tris);
    rast.free();
    const auto outputPhase = clock.now() - start;
    start = clock.now();

    std::clog << "[Timing] Input phase: " << (int)(inputPhase.count() / 1e6) << " ms" << std::endl;
    std::clog << "         Rasterize phase: " << (int)(rasterizePhase.count() / 1e6) << " ms" << std::endl;
    std::clog << "         Output phase: " << (int)(outputPhase.count() / 1e6) << " ms" << std::endl;
    return 0;
}
