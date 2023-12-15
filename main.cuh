#include <assert.h>
#include "grid.cuh"
#include "geometry.cuh"
constexpr const int TILE_SIZE = 8;
constexpr const int NTHREAD_1D = 512;

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

__global__ void volume_scan_kernel(const float * tris, float * grid, const int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < N && y < N && z < N)
    {
    }
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

__global__ void rasterize_fill_kernel(const float val, const int L, float * outGridDist)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L) return;
    outGridDist[g] = val;
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

void volume_scan_cuda(const float * tris, float * grid, const int N)
{
    dim3 dimBlock(ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE));
    dim3 dimGrid(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    volume_scan_kernel<<<dimBlock, dimGrid>>>(tris, grid, N);
}

float * rasterize_tris(const float3 * tris, const int F, const int R, const float band)
{
    uint * idx;
    uint * grid;
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
    cudaMallocManaged(&outIdx, *totalSize * sizeof(uint));
    cudaMallocManaged(&outGrid, *totalSize * sizeof(uint));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
    cudaDeviceSynchronize();

    float * gridDist = nullptr;
    assert(CUDA_SUCCESS == cudaMallocManaged(&gridDist, R * R * R * sizeof(float)));
    rasterize_fill_kernel<<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        1e9f, R * R * R, gridDist
    );
    cudaDeviceSynchronize();
    rasterize_reduce_kernel<<<ceil_div(*totalSize, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, *totalSize, R, gridDist
    );
    
    cudaFree(idx);
    cudaFree(grid);
    cudaFree(totalSize);
    cudaFree(tempBlockOffset);
    cudaFree(outIdx);
    cudaFree(outGrid);

    return gridDist;
}
