#include <assert.h>
#include <float.h>
#include <utility>
#include "grid.cuh"
#include "geometry.cuh"
constexpr const int TILE_SIZE = 8;
constexpr const int NTHREAD_1D = 512;

#define CHECK_CUDA(ans) _cuda_check<true>((ans), __FILE__, __LINE__)
template<bool abort>
inline void _cuda_check(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if constexpr (abort) exit(code);
   }
}

struct RasterizeResult
{
    float * gridDist;
    float3 * gridPseudoNormal;
    int * gridRepPoint;

    void free()
    {
        cudaFree(gridDist);
        cudaFree(gridPseudoNormal);
        cudaFree(gridRepPoint);
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

__forceinline__ __device__ uint shuffler(uint v, uint bmask)
{
    return v;
    // return (v * (v + 1) / 2) & bmask;
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
    const uint b = blockIdx.x;
    const long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
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
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    
    const float thresh = 0.87 / N + band;
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

template<typename T>
__global__ void common_fill_kernel(const T val, const uint L, T * outGrid)
{
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L) return;
    outGrid[g] = val;
}

__global__ void common_arange_kernel(uint * t, uint limit)
{
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= limit) return;
    t[g] = g;
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
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
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
    const float * gridDist, float3 * outGridPseudoNormal, int * outGridRepIdx
) {
    const uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= M) return;
    const uint3 nxyz = unpack_id(grid[g]);
    const float3 fxyz = (make_float3(nxyz.x, nxyz.y, nxyz.z) + 0.5f) / (float)N;
    const uint access = to_gidx(nxyz, N);
    
    const uint tofs = idx[g];
    const float3 v1 = tris[tofs * 3];
    const float3 v2 = tris[tofs * 3 + 1];
    const float3 v3 = tris[tofs * 3 + 2];

    const float cmp = gridDist[access] + FLT_EPSILON;
    if (sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz)) < cmp)
    {
        // TODO: pseudo-normal for vertices? how to handle float-point errors?
        // https://dl.acm.org/doi/pdf/10.5555/2619648.2619655
        // Signed Distance Fields for Polygon Soup Meshes
        // https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B%C3%A6rentzen.pdf
        // Signed distance computation using the angle weighted pseudonormal
        const float3 n = normalize(cross(v2 - v1, v3 - v1));
        atomicAdd(&outGridPseudoNormal[access].x, n.x);
        atomicAdd(&outGridPseudoNormal[access].y, n.y);
        atomicAdd(&outGridPseudoNormal[access].z, n.z);
        uint pt = tofs * 3;
        if (sqrt(point_to_segment_dist_sqr(v2, v3, fxyz)) < cmp)
            pt = tofs * 3 + 1;
        if (length(v3 - fxyz) < cmp)
            pt = tofs * 3 + 2;
        atomicMax(&outGridRepIdx[access], pt);
    }
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
    common_fill_kernel<uint><<<ceil_div(F, NTHREAD_1D), NTHREAD_1D>>>(startId, F, grid);

    // layer a
    rasterize_layer_kernel<true><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, totalSize, nullptr, nullptr
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    const uint las = *totalSize;
    CHECK_CUDA(cudaMallocManaged(&outIdx, las * sizeof(uint)));
    CHECK_CUDA(cudaMallocManaged(&outGrid, las * sizeof(uint)));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, La, F, La, band, tempBlockOffset, nullptr, outIdx, outGrid
    );
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
    CHECK_CUDA(cudaDeviceSynchronize());
    const uint lbs = *totalSize;
    CHECK_CUDA(cudaMallocManaged(&outIdx, lbs * sizeof(uint)));
    CHECK_CUDA(cudaMallocManaged(&outGrid, lbs * sizeof(uint)));
    rasterize_layer_kernel<false><<<blocks, NTHREAD_1D>>>(
        tris, idx, grid, Lb, las, R, band, tempBlockOffset, nullptr, outIdx, outGrid
    );

    RasterizeResult rasterizeResult;
    CHECK_CUDA(cudaMallocManaged(&rasterizeResult.gridDist, R * R * R * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&rasterizeResult.gridPseudoNormal, R * R * R * sizeof(float3)));
    CHECK_CUDA(cudaMallocManaged(&rasterizeResult.gridRepPoint, R * R * R * sizeof(int)));
    common_fill_kernel<float><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        1e9f, R * R * R, rasterizeResult.gridDist
    );
    common_fill_kernel<float3><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        make_float3(0, 0, 0), R * R * R, rasterizeResult.gridPseudoNormal
    );
    common_fill_kernel<int><<<ceil_div(R * R * R, NTHREAD_1D), NTHREAD_1D>>>(
        -1, R * R * R, rasterizeResult.gridRepPoint
    );
    rasterize_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R, rasterizeResult.gridDist
    );
    rasterize_arg_reduce_kernel<<<ceil_div(lbs, NTHREAD_1D), NTHREAD_1D>>>(
        tris, outIdx, outGrid, lbs, R,
        rasterizeResult.gridDist, rasterizeResult.gridPseudoNormal, rasterizeResult.gridRepPoint
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(idx));
    CHECK_CUDA(cudaFree(grid));
    CHECK_CUDA(cudaFree(totalSize));
    CHECK_CUDA(cudaFree(tempBlockOffset));
    CHECK_CUDA(cudaFree(outIdx));
    CHECK_CUDA(cudaFree(outGrid));

    return rasterizeResult;
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

__forceinline__ __device__ void cts_atomic_union(uint * __restrict__ parents, uint x, uint y)
{
    // REVIEW: I didn't use rank or size based union
    // Because I cannot implement it without locks
    // And we can expect xor-shuffled performance to be the same without
    while (true)
    {
        x = cts_find(parents, x);
        y = cts_find(parents, y);
        if (x == y)
            return;
        atomicCAS(&parents[max(x, y)], max(x, y), min(x, y));
    }
}

__global__ void volume_cts_kernel(const float3 * tris, const RasterizeResult rast, uint * parents, const uint N, const int shfBitmask)
{
    const uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);
    const uint shfm = shuffler(access, shfBitmask);
    const uint shfex = shuffler(N * N * N, shfBitmask);
    const int dist = rast.gridDist[access];

    if (dist > 2.0f / N)
    {
        if (xyz.x == 0 || xyz.y == 0 || xyz.z == 0)
           cts_atomic_union(parents, shfm, shfex);
        else
        {
            const uint3 check[3] = {
                xyz - make_uint3(1, 0, 0),
                xyz - make_uint3(0, 1, 0),
                xyz - make_uint3(0, 0, 1),
            };
            #pragma unroll
            for (int s = 0; s < 3; s++)
            {
                const uint naccess = to_gidx(check[s], N);
                if (rast.gridDist[naccess] > 2.0f / N)
                    cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }
        }
    }
    else
    {
        const uint3 check[6] = {
            xyz - make_uint3(1, 0, 0),
            xyz - make_uint3(0, 1, 0),
            xyz - make_uint3(0, 0, 1),
            xyz + make_uint3(1, 0, 0),
            xyz + make_uint3(0, 1, 0),
            xyz + make_uint3(0, 0, 1),
        };
        const float3 c0 = (i2f(xyz) + 0.5f) / (float)N;
        const float3 v0 = tris[rast.gridRepPoint[access]];
        const float3 n0 = rast.gridPseudoNormal[access];
        #pragma unroll
        for (int s = 0; s < 6; s++)
        {
            const uint3 z1 = clamp(check[s], 0, N - 1);
            const float3 c1 = (i2f(check[s]) + 0.5f) / (float)N;
            const uint naccess = to_gidx(z1, N);
            if (rast.gridRepPoint[naccess] == -1)
            {
                if (!(dot(c0 - v0, n0) * dot(c1 - v0, n0) > 0))
                    continue;
                cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }
            else
            {
                const float3 v1 = tris[rast.gridRepPoint[naccess]];
                const float3 n1 = rast.gridPseudoNormal[naccess];
                if (!(dot(c0 - v0, n0) * dot(c1 - v1, n1) > 0))
                    continue;
                cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }
        }
    }
}

__global__ void volume_apply_sign_kernel(RasterizeResult rast, const uint * parents, const int N, const int shfBitmask)
{
    const uint root = cts_find(parents, shuffler(N * N * N, shfBitmask));
    
    const uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);
    const uint shfm = shuffler(access, shfBitmask);

    if (cts_find(parents, shfm) != root)
        rast.gridDist[access] *= -1;
}

__global__ void volume_bellman_ford_kernel(const float * tris, RasterizeResult rast, bool * gridIsEx, bool * globalChanged, const int N)
{
    // alternative algorithm to union-find
    uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    uint access = to_gidx(xyz, N);
    int changed = 0;
    #pragma unroll
    for (int it = 0; it < 16; it++)
    {

    }
    changed = __syncthreads_or(changed);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && changed) *globalChanged = true;
}

inline uint npo2(const uint n)
{
    uint v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void fill_signs(const float3 * tris, const int N, RasterizeResult rast)
{
    dim3 dimBlock(ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE), ceil_div(N, TILE_SIZE));
    dim3 dimGrid(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    // cudaFuncSetCacheConfig(volume_bellman_ford_kernel, cudaFuncCachePreferL1);
    // volume_bellman_ford_kernel<<<dimBlock, dimGrid>>>(tris, rast, nullptr, N);
    uint * parents;
    const uint nodeCount = npo2(N * N * N + 1);
    const uint shfBitmask = npo2(N * N * N + 1) - 1;
    CHECK_CUDA(cudaMallocManaged(&parents, npo2(N * N * N + 1) * sizeof(uint)));

    CHECK_CUDA(cudaFuncSetCacheConfig(common_arange_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(volume_cts_kernel, cudaFuncCachePreferL1));
    CHECK_CUDA(cudaFuncSetCacheConfig(volume_apply_sign_kernel, cudaFuncCachePreferL1));

    common_arange_kernel<<<ceil_div(nodeCount, NTHREAD_1D), NTHREAD_1D>>>(parents, nodeCount);
    CHECK_CUDA(cudaDeviceSynchronize());
    volume_cts_kernel<<<dimBlock, dimGrid>>>(tris, rast, parents, N, shfBitmask);
    CHECK_CUDA(cudaDeviceSynchronize());
    volume_apply_sign_kernel<<<dimBlock, dimGrid>>>(rast, parents, N, shfBitmask);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(parents));
}
