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
    if (!fi)
    {
        std::cerr << "[Error] Cannot open input file" << std::endl;
        return 1;
    }
    int F;
    fi >> F;
    // copies all data into buffer
    float3 * tris;
    cudaMallocManaged(&tris, F * 3 * sizeof(float3));

    for (int i = 0; i < F; i++)
        for (int j = 0; j < 3; j++)
            fi >> tris[3 * i + j].x >> tris[3 * i + j].y >> tris[3 * i + j].z;
    fi.close();
    const auto inputPhase = clock.now() - start;
    start = clock.now();

    // for (int i = 0; i < buffer.size() / 3 / sizeof(float); i++)
    //     printf("%.2f %.2f %.2f\n", tris[i].x, tris[i].y, tris[i].z);

    constexpr const int N = 128;
    RasterizeResult rast = rasterize_tris(tris, F, N, 4.0f / N);
    const auto rasterizePhase = clock.now() - start;
    start = clock.now();
    fill_signs(tris, N, rast);
    const auto signPhase = clock.now() - start;
    start = clock.now();
    static char _output_buf[2 * 1024 * 1024];
    std::ofstream fo("output.txt", std::fstream::trunc);
    fo.rdbuf()->pubsetbuf(_output_buf, 2 * 1024 * 1024);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
                fo << rast.gridDist[i * N * N + j * N + k] << '\t';
            fo << '\n';
        }
    fo.close();
    cudaFree(tris);
    rast.free();
    const auto outputPhase = clock.now() - start;
    start = clock.now();

    std::clog << "[Timing] Input phase: " << (int)(inputPhase.count() / 1e6) << " ms" << std::endl;
    std::clog << "         Rasterize phase: " << (int)(rasterizePhase.count() / 1e6) << " ms" << std::endl;
    std::clog << "         Sign phase: " << (int)(signPhase.count() / 1e6) << " ms" << std::endl;
    std::clog << "         Output phase: " << (int)(outputPhase.count() / 1e6) << " ms" << std::endl;
    return 0;
}
