#pragma once
#include "structs.cuh"
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"

constexpr const int KERNEL_ITER = 8;
constexpr const int CPU_ITER = 1;

__forceinline__ __device__ bool vertex_relax(const float3 * tris, const RasterizeResult rast, const int3 xyz, char * state, const uint N)
{
    const uint access = to_gidx(make_uint3(xyz), N);
    const float dist = rast.gridDist[access];
    if (dist < 0.86603f / N) return false;
    // state[access] is 2.

    const int3 check[26] = {
        xyz - make_int3(1, 0, 0),
        xyz - make_int3(0, 1, 0),
        xyz - make_int3(0, 0, 1),

        xyz + make_int3(1, 0, 0),
        xyz + make_int3(0, 1, 0),
        xyz + make_int3(0, 0, 1),

        xyz + make_int3(1, -1, 0),
        xyz + make_int3(0, 1, -1),
        xyz + make_int3(-1, 0, 1),
        xyz + make_int3(1, 1, 0),
        xyz + make_int3(0, 1, 1),
        xyz + make_int3(1, 0, 1),
        xyz + make_int3(-1, 1, 0),
        xyz + make_int3(0, -1, 1),
        xyz + make_int3(1, 0, -1),
        xyz - make_int3(1, 1, 0),
        xyz - make_int3(0, 1, 1),
        xyz - make_int3(1, 0, 1),

        xyz + make_int3(-1, -1, -1),
        xyz + make_int3(-1, -1, 1),
        xyz + make_int3(-1, 1, -1),
        xyz + make_int3(-1, 1, 1),
        xyz + make_int3(1, -1, -1),
        xyz + make_int3(1, -1, 1),
        xyz + make_int3(1, 1, -1),
        xyz + make_int3(1, 1, 1),
    };

    const float3 c0 = (i2f(make_uint3(xyz)) + 0.5f) / (float)N;
    const int tidx = rast.gridIdx[access];
    const int tofs = max(tidx, 0) * 3;
    const float3 v01 = tris[tofs];
    const float3 v02 = tris[tofs + 1];
    const float3 v03 = tris[tofs + 2];
    const float3 p0 = closest_point_on_triangle_to_point(v01, v02, v03, c0);
    
    bool changed = false;

    #pragma unroll
    for (int s = 0; s < 26; s++)
    {
        const uint3 z1 = make_uint3(clamp(check[s], 0, N - 1));
        const uint naccess = to_gidx(z1, N);
        if (state[naccess] != 0) continue;
        if (rast.gridDist[naccess] >= 0.86603f / N)
        {
            state[naccess] = 2;
            changed = true;
        }
        else
        {
            assert(tidx >= 0);
            const float3 c1 = (i2f(z1) + 0.5f) / (float)N;
            const int nidx = rast.gridIdx[naccess] * 3;
            const float3 v11 = tris[nidx];
            const float3 v12 = tris[nidx + 1];
            const float3 v13 = tris[nidx + 2];
            const float3 p1 = closest_point_on_triangle_to_point(v11, v12, v13, c1);
            
            if (!(dot(normalize(c0 - p0), normalize(c1 - p1)) > 0))
                continue;
            if (!(dot(normalize(c0 - p1), normalize(c1 - p1)) > 0))
                continue;
            if (!(dot(normalize(c0 - p0), normalize(c1 - p0)) > 0))
                continue;
            state[naccess] = 2;
            changed = true;
        }
    }
    return changed;
}

__global__ void volume_bellman_ford_kernel(const float3 * tris, const RasterizeResult rast, char * state, const uint N, bool * globalChanged)
{
    bool changed = false;
    const uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);
    if (xyz.x == 0 || xyz.y == 0 || xyz.z == 0 || xyz.x == N - 1 || xyz.y == N - 1 || xyz.z == N - 1)
    {
        if (state[access] == 0 && rast.gridDist[access] >= 0.86603f / N)
        {
            state[access] = 2;
            changed = true;
        }
    }
    for (int it = 0; it < KERNEL_ITER; it++)
    {
        if (state[access] != 2) continue;
        changed |= vertex_relax(tris, rast, make_int3(xyz), state, N);
        state[access] = 1;
    }
    changed = __syncthreads_or((int)changed) != 0;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && changed)
        *globalChanged = true;
}

__global__ void volume_apply_sign_kernel(RasterizeResult rast, const char * state, const int N)
{
    const uint3 xyz = blockIdx * blockDim + threadIdx;
    if (xyz.x >= N || xyz.y >= N || xyz.z >= N) return;
    const uint access = to_gidx(xyz, N);

    if (state[access] == 0)
        rast.gridDist[access] *= -1;
}
