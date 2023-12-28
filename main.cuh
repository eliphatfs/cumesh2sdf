#pragma once
#include "structs.cuh"
#include "grid.cuh"
#include "geometry.cuh"
#include "commons.cuh"
#include "rasterize.cuh"
#include "sign.cuh"
constexpr const int TILE_SIZE = 8;
constexpr const int NTHREAD_1D = 512;

constexpr int ceil_div(const int a, const int b)
{
    return ((a) + (b) - 1) / (b);
}

RasterizeResult rasterize_tris(const float3 * tris, const int F, const int R, const float band)
{
    uint * idx;
    uint * grid;
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<true>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<false>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_reduce_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_arg_reduce_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaMallocManaged(&idx, F * sizeof(uint)));
    CHECK_CUDA(cudaMallocManaged(&grid, F * sizeof(uint)));

    uint startId = pack_id(make_uint3(0, 0, 0));

    uint * tempBlockOffset;
    uint * totalSize;
    uint * outIdx;
    uint * outGrid;

    const uint La = R >= 256 ? 16 : 8;
    const uint Lb = R / La;
    assert(R % La == 0);
    CHECK_CUDA(cudaMallocManaged(&totalSize, sizeof(uint)));
    *totalSize = 0;
    uint blocks = ceil_div(La * La * La * F, NTHREAD_1D);
    CHECK_CUDA(cudaMallocManaged(&tempBlockOffset, blocks * sizeof(uint)));

    common_arange_kernel<<<ceil_div(F, NTHREAD_1D), NTHREAD_1D>>>(idx, F);
    CHECK_CUDA(cudaGetLastError());
    common_fill_kernel<uint><<<ceil_div(F, NTHREAD_1D), NTHREAD_1D>>>(startId, F, grid);
    CHECK_CUDA(cudaGetLastError());

    // layer a
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    const uint las = *totalSize;
    CHECK_CUDA(cudaMallocManaged(&outIdx, las * sizeof(uint)));
    CHECK_CUDA(cudaMallocManaged(&outGrid, las * sizeof(uint)));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(idx));
    CHECK_CUDA(cudaFree(grid));
    CHECK_CUDA(cudaFree(tempBlockOffset));
    idx = outIdx;
    grid = outGrid;

    // layer b
    // assert((long long)Lb * (long long)Lb * (long long)Lb * (long long)las < 4294967295u);
    blocks = ceil_div(Lb * Lb * Lb * las, NTHREAD_1D);
    *totalSize = 0;
    CHECK_CUDA(cudaMallocManaged(&tempBlockOffset, blocks * sizeof(uint)));
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    const uint lbs = *totalSize;
    CHECK_CUDA(cudaMallocManaged(&outIdx, lbs * sizeof(uint)));
    CHECK_CUDA(cudaMallocManaged(&outGrid, lbs * sizeof(uint)));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
    CHECK_CUDA(cudaGetLastError());

    RasterizeResult rasterizeResult;
    CHECK_CUDA(cudaMallocManaged(&rasterizeResult.gridDist, R * R * R * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&rasterizeResult.gridIdx, R * R * R * sizeof(int)));
    common_fill_kernel<float><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        1e9f, R * R * R, rasterizeResult.gridDist
    );
    CHECK_CUDA(cudaGetLastError());
    common_fill_kernel<int><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        -1, R * R * R, rasterizeResult.gridIdx
    );
    CHECK_CUDA(cudaGetLastError());
    rasterize_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R, rasterizeResult.gridDist
    );
    CHECK_CUDA(cudaGetLastError());
    rasterize_arg_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R,
        rasterizeResult.gridDist, rasterizeResult.gridIdx
    );
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(idx));
    CHECK_CUDA(cudaFree(grid));
    CHECK_CUDA(cudaFree(totalSize));
    CHECK_CUDA(cudaFree(tempBlockOffset));
    CHECK_CUDA(cudaFree(outIdx));
    CHECK_CUDA(cudaFree(outGrid));

    return rasterizeResult;
}

void fill_signs(const float3 * tris, const int N, RasterizeResult rast)
{
    dim3 dimBlock(ceil_div(N, 16), ceil_div(N, 16), ceil_div(N, 1));
    dim3 dimGrid(16, 16, 1);

    char * state;
    bool * changed;
    CHECK_CUDA(cudaMallocManaged(&state, N * N * N * sizeof(char)));
    CHECK_CUDA(cudaMallocManaged(&changed, sizeof(bool)));
    *changed = true;

    CHECK_CUDA(cudaFuncSetCacheConfig(volume_bellman_ford_kernel, cudaFuncCachePreferL1));
    common_fill_kernel<char><<<ceil_div(N * N * N, NTHREAD_1D), NTHREAD_1D>>>(0, N * N * N, state);
    CHECK_CUDA(cudaDeviceSynchronize());

    while (*changed)
    {
        *changed = false;
        for (int it = 0; it < CPU_ITER; it++)
        {
            volume_bellman_ford_kernel<<<dimBlock, dimGrid>>>(tris, rast, state, N, changed);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(changed));
    volume_apply_sign_kernel<<<dimBlock, dimGrid>>>(rast, state, N);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(state));
}
