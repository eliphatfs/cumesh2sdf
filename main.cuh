#include <assert.h>
#include <utility>
#include "grid.cuh"
#include "geometry.cuh"
constexpr const int TILE_SIZE = 8;
constexpr const int NTHREAD_1D = 512;

struct RasterizeResult
{
    float * gridDist;
    int * gridIdx;

    void free()
    {
        cudaFree(gridDist);
        cudaFree(gridIdx);
    }
};

constexpr int ceil_div(const int a, const int b)
{
    return ((a) + (b) - 1) / (b);
}

__forceinline__ __device__ uint shash(uint x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__forceinline__ __device__ uint chash(uint seed, uint value)
{
    // formula from boost
    return value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

__forceinline__ __device__ uint vhash3(uint3 v)
{
    return chash(chash(shash(v.x), shash(v.y)), shash(v.z));
}

template<bool probe>
__global__ void rasterize_layer_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const uint S, const int M, const int N, const float band,
    uint * tempBlockOffset, uint * totalSize, uint * outIdx, uint * outGrid
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // S subdiv
    // N scale of current grid, pre-multiplied by S
    const int b = blockIdx.x;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= S * S * S * M) return;

    __shared__ uint blockSize;

    if (threadIdx.x == 0) blockSize = 0;
    __syncthreads();

    const int i = g % S;
    const int j = (g / S) % S;
    const int k = (g / (S * S)) % S;
    const int t = g / (S * S * S);

    const uint tofs = idx[t];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const uint gid = grid[t];
    const uint3 nxyz = unpack_id(gid) * S + make_uint3(i, j, k);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    
    const float thresh = 0.87 / N + band;
    const bool intersect = point_to_tri_dist_sqr(v1, v2, v3, fxyz) < thresh * thresh;
    
    if (intersect)
    {
        uint inblock = atomicAdd(&blockSize, 1);
        if constexpr (!probe)
        {
            outIdx[tempBlockOffset[b] + inblock] = tofs;
            outGrid[tempBlockOffset[b] + inblock] = pack_id(nxyz);
        }
    }
    __syncthreads();
    if constexpr (probe)
    {
        if (threadIdx.x == 0)
        {
            tempBlockOffset[b] = atomicAdd(totalSize, blockSize);
        }
    }
}

__device__ __forceinline__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void rasterize_fill_kernel(const float val, const int L, float * outGrid)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L) return;
    outGrid[g] = val;
}

__global__ void rasterize_fill_kernel(const int val, const int L, int * outGrid)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L) return;
    outGrid[g] = val;
}

__global__ void rasterize_reduce_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const int M, const int N,
    float * outGridDist
) {
    // idx [M] index into tris
    // tris [?, 3]
    // grid [M] packed grid position
    // N size of target grid
    // outGridDist: [N, N, N] distance
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M) return;
    const uint3 nxyz = unpack_id(grid[g]);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    const uint access = to_gidx(nxyz, N);
    
    const uint tofs = idx[g];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    atomicMin(outGridDist + access, sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz)));
}

__global__ void rasterize_arg_reduce_kernel(
    const float3 * tris, const uint * idx, const uint * grid, const int M, const int N,
    const float * gridDist, int * outGridIdx
) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M) return;
    const uint3 nxyz = unpack_id(grid[g]);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    const uint access = to_gidx(nxyz, N);
    
    const uint tofs = idx[g];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    if (sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz)) == gridDist[access])
        atomicMax(outGridIdx + access, tofs);
}

__forceinline__ __device__ uint cts_find(const uint * parents, const uint i)
{
    // REVIEW: we shall not use path compression?
    // Because parallelism is good without atomic writes.
    uint c = i;
    while (parents[c] != c)
        c = parents[c];
    return c;
}

__forceinline__ __device__ void cts_atomic_union(uint * parents, uint x, uint y)
{
    while (true)
    {
        x = cts_find(parents, x);
        y = cts_find(parents, y);
        if (x == y) return;
        if (atomicMin(&parents[y], x) == y)
            return;
    }
}

RasterizeResult rasterize_tris(const float3 * tris, const int F, const int R, const float band)
{
    uint * idx;
    uint * grid;
    cudaFuncSetCacheConfig(rasterize_layer_kernel<true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(rasterize_layer_kernel<false>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(rasterize_reduce_kernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(rasterize_arg_reduce_kernel, cudaFuncCachePreferL1);
    cudaMallocManaged(&idx, F * sizeof(uint));
    cudaMallocManaged(&grid, sizeof(uint));

    for (uint i = 0; i < F; i++)
        idx[i] = i;
    grid[0] = pack_id(make_uint3(0, 0, 0));

    uint * tempBlockOffset;
    uint * totalSize;
    uint * outIdx;
    uint * outGrid;

    const uint La = R >= 256 ? 16 : 8;
    const uint Lb = R / La;
    assert(R % La == 0);
    cudaMallocManaged(&totalSize, sizeof(uint));
    *totalSize = 0;
    uint blocks = ceil_div(La * La * La * F, NTHREAD_1D);
    cudaMallocManaged(&tempBlockOffset, blocks * sizeof(uint));

    // layer a
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    const uint las = *totalSize;
    cudaMallocManaged(&outIdx, las * sizeof(uint));
    cudaMallocManaged(&outGrid, las * sizeof(uint));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
    cudaDeviceSynchronize();

    cudaFree(idx);
    cudaFree(grid);
    cudaFree(tempBlockOffset);
    idx = outIdx;
    grid = outGrid;

    // layer b
    blocks = ceil_div(Lb * Lb * Lb * las, NTHREAD_1D);
    *totalSize = 0;
    cudaMallocManaged(&tempBlockOffset, blocks * sizeof(uint));
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    cudaDeviceSynchronize();
    const uint lbs = *totalSize;
    cudaMallocManaged(&outIdx, lbs * sizeof(uint));
    cudaMallocManaged(&outGrid, lbs * sizeof(uint));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, nullptr, outIdx, outGrid
    );

    RasterizeResult rasterizeResult;
    cudaMallocManaged(&rasterizeResult.gridDist, R * R * R * sizeof(float));
    cudaMallocManaged(&rasterizeResult.gridIdx, R * R * R * sizeof(int));
    rasterize_fill_kernel<<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        1e9f, R * R * R, rasterizeResult.gridDist
    );
    rasterize_fill_kernel<<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        -1, R * R * R, rasterizeResult.gridIdx
    );
    rasterize_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R, rasterizeResult.gridDist
    );
    rasterize_arg_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R, rasterizeResult.gridDist, rasterizeResult.gridIdx
    );
    
    cudaFree(idx);
    cudaFree(grid);
    cudaFree(totalSize);
    cudaFree(tempBlockOffset);
    cudaFree(outIdx);
    cudaFree(outGrid);

    return rasterizeResult;
}

__global__ void volume_bellman_ford_kernel(const float * tris, RasterizeResult rast, bool * gridIsEx, const int N)
{
    uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    int access = to_gidx(xyz, N);
    while (true)
    {
        bool changed = false;
        #pragma unroll
        for (int it = 0; it < 16; it++)
        {

        }
    }
}

void find_signs_cuda(const float * tris, RasterizeResult rast, const int N)
{
    dim3 dimBlock(ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE));
    dim3 dimGrid(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    volume_bellman_ford_kernel<<<dimBlock, dimGrid>>>(tris, rast, nullptr, N);
}