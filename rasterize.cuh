#pragma once
#include "structs.cuh"
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"
#include "allocator.cuh"
#include <utility>
#include <vector>
#include <iostream>


template<bool probe>
__global__ void rasterize_layer_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const uint S, const int M, const int N, const float band,
    uint * __restrict__ tempBlockOffset, uint * __restrict__ totalSize,
    uint * __restrict__ outIdx, uint * __restrict__ outGrid
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // S subdiv
    // N scale of current grid, pre-multiplied by S
    const uint b = blockIdx.x;
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= S * S * S * M) return;

    __shared__ uint blockSize;

    if (threadIdx.x == 0) blockSize = 0;
    __syncthreads();

    const int mo = g % (S * S * S);
    const int i = mo % S;
    const int j = (mo / S) % S;
    const int k = (mo / (S * S)) % S;
    const int t = g / (S * S * S);

    const uint tofs = idx[t];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const uint gid = grid[t];
    const uint3 nxyz = unpack_id(gid) * S + make_uint3(i, j, k);
    const float3 fxyz = (i2f(nxyz) + 0.5f) / (float)N;
    
    const float thresh = 0.87f / N + band;
    const bool intersect = point_to_tri_dist_sqr(v1, v2, v3, fxyz) < thresh * thresh;
    
    if (intersect)
    {
        uint inblock = atomicAdd(&blockSize, 1);
        if constexpr (!probe)
        {
            const uint bofs = tempBlockOffset[b];
            outIdx[bofs + inblock] = tofs;
            outGrid[bofs + inblock] = pack_id(nxyz);
        }
    }
    if constexpr (probe)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            tempBlockOffset[b] = atomicAdd(totalSize, blockSize);
        }
    }
}

__global__ void rasterize_reduce_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const int M, const int N,
    float * __restrict__ outGridDist, bool * __restrict__ outGridCollide
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // N size of target grid
    // outGridDist: [N, N, N] distance
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M) return;
    const uint3 nxyz = unpack_id(grid[g]);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    const uint access = to_gidx(nxyz, N);
    
    const uint tofs = idx[g];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const float finalDist = sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz));
    atomicMin(outGridDist + access, finalDist);

    const float rayth = 1.0f / N + FLT_EPSILON;
    if (finalDist > rayth) return;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(1, 0, 0), finalDist) <= rayth)
        outGridCollide[access * 3] = true;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(0, 1, 0), finalDist) <= rayth)
        outGridCollide[access * 3 + 1] = true;
    if (ray_triangle_hit_dist(v1, v2, v3, fxyz, make_float3(0, 0, 1), finalDist) <= rayth)
        outGridCollide[access * 3 + 2] = true;
}

inline uint rasterize_layer_internal(const float3 * tris, const int S, const int M, const int N, uint ** pidx, uint ** pgrid, uint * totalSize, const float band, MemoryAllocator& ma)
{
    uint * idx = *pidx;
    uint * grid = *pgrid;
    
    if (S * S * S * M != S * S * S * (long long)M)
    {
        std::cerr << "Warning: Overflow in rasterize layer. Please reduce batch size." << std::endl;
    }
    uint blocks = ceil_div(S * S * S * M, NTHREAD_1D);
    uint * tempBlockOffset = ma.alloc<uint>(blocks);

    // probe
    common_fill_kernel<uint><<<1, 1>>>(0, 1, totalSize);
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, S, M, N, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    CHECK_CUDA(cudaGetLastError());

    uint las;
    CHECK_CUDA(cudaMemcpy(&las, totalSize, sizeof(uint), cudaMemcpyDeviceToHost));

    uint * outIdx = ma.alloc<uint>(las);
    uint * outGrid = ma.alloc<uint>(las);

    // fill
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, S, M, N, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
    CHECK_CUDA(cudaGetLastError());

    ma.free(idx);
    ma.free(grid);
    ma.free(tempBlockOffset);

    *pidx = outIdx;
    *pgrid = outGrid;
    return las;
}

inline RasterizeResult rasterize_tris_internal(const float3 * tris, const int F, const std::vector<int> SS, const int B, const float band)
{
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<true>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_layer_kernel<false>, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(rasterize_reduce_kernel, cudaFuncCachePreferL1));

    int N = 1;
    for (uint i = 0; i < SS.size(); i++)
        N *= SS[i];
    const int R = N;

    RasterizeResult rasterizeResult;
    CHECK_CUDA(cudaMalloc(&rasterizeResult.gridDist, R * R * R * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rasterizeResult.gridCollide, R * R * R * sizeof(bool) * 3));

    uint * totalSize;
    CHECK_CUDA(cudaMalloc(&totalSize, sizeof(uint)));

    MemoryAllocator ma;

    uint startId = pack_id(make_uint3(0, 0, 0));
    int M;
    for (int j = 0; j < F; j += B)
    {
        N = 1;
        M = min(B, F - j);

        uint * idx = ma.alloc<uint>(M);
        uint * grid = ma.alloc<uint>(M);
        
        common_arange_kernel<<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(idx, M, j);
        CHECK_CUDA(cudaGetLastError());
        common_fill_kernel<uint><<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(startId, M, grid);
        CHECK_CUDA(cudaGetLastError());

        for (uint i = 0; i < SS.size(); i++)
        {
            N *= SS[i];
            M = rasterize_layer_internal(tris, SS[i], M, N, &idx, &grid, totalSize, band, ma);
        }

        if (j == 0)
        {
            common_fill_kernel<float><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
                1e9f, R * R * R, rasterizeResult.gridDist
            );
            CHECK_CUDA(cudaGetLastError());
            common_fill_kernel<bool><<<ceil_div(R * R * R * 3LL, NTHREAD_1D), NTHREAD_1D>>>(
                false, R * R * R * 3, rasterizeResult.gridCollide
            );
            CHECK_CUDA(cudaGetLastError());
        }
        rasterize_reduce_kernel<<<ceil_div(M, NTHREAD_1D), NTHREAD_1D>>>(
            tris, idx, grid, M, R, rasterizeResult.gridDist, rasterizeResult.gridCollide
        );
        CHECK_CUDA(cudaGetLastError());
        ma.free(idx);
        ma.free(grid);
    }
    CHECK_CUDA(cudaFree(totalSize));
    CHECK_CUDA(cudaDeviceSynchronize());

    return rasterizeResult;
}

static RasterizeResult rasterize_tris(const float3 * tris, const int F, const int R, const float band, const int B = 131072)
{
    assert(R <= 1024);
    std::vector<int> s;
    int N = R;
    while (N > 4)
    {
        s.push_back(4);
        N /= 4;
    }
    s.push_back(N);
    return rasterize_tris_internal(tris, F, s, B, band);
}
