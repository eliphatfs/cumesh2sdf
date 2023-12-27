#pragma once
#include "structs.cuh"
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"

__forceinline__ __device__ uint shuffler(uint v, uint bmask)
{
    return (v * (v + 1) / 2) & bmask;
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

    if (dist >= 0.86603f / N)
    {
        if (xyz.x == 0 || xyz.y == 0 || xyz.z == 0)
           cts_atomic_union(parents, shfm, shfex);
        else
        {
            const uint3 check[7] = {
                xyz - make_uint3(1, 0, 0),
                xyz - make_uint3(0, 1, 0),
                xyz - make_uint3(0, 0, 1),

                xyz - make_uint3(1, 1, 0),
                xyz - make_uint3(0, 1, 1),
                xyz - make_uint3(1, 0, 1),

                xyz - make_uint3(1, 1, 1),
            };
            #pragma unroll
            for (int s = 0; s < 7; s++)
            {
                const uint naccess = to_gidx(check[s], N);
                if (rast.gridDist[naccess] >= 0.86603f / N)
                    cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }
        }
    }
    else
    {
        const int3 ixyz = make_int3(xyz);
        const int3 check[26] = {
            ixyz - make_int3(1, 0, 0),
            ixyz - make_int3(0, 1, 0),
            ixyz - make_int3(0, 0, 1),

            ixyz + make_int3(1, 0, 0),
            ixyz + make_int3(0, 1, 0),
            ixyz + make_int3(0, 0, 1),

            ixyz + make_int3(1, -1, 0),
            ixyz + make_int3(0, 1, -1),
            ixyz + make_int3(-1, 0, 1),
            ixyz + make_int3(1, 1, 0),
            ixyz + make_int3(0, 1, 1),
            ixyz + make_int3(1, 0, 1),
            ixyz + make_int3(-1, 1, 0),
            ixyz + make_int3(0, -1, 1),
            ixyz + make_int3(1, 0, -1),
            ixyz - make_int3(1, 1, 0),
            ixyz - make_int3(0, 1, 1),
            ixyz - make_int3(1, 0, 1),

            ixyz + make_int3(-1, -1, -1),
            ixyz + make_int3(-1, -1, 1),
            ixyz + make_int3(-1, 1, -1),
            ixyz + make_int3(-1, 1, 1),
            ixyz + make_int3(1, -1, -1),
            ixyz + make_int3(1, -1, 1),
            ixyz + make_int3(1, 1, -1),
            ixyz + make_int3(1, 1, 1),
        };
        const float3 c0 = (i2f(xyz) + 0.5f) / (float)N;
        // const float3 v0 = tris[rast.gridRepPoint[access]];
        const float3 v1 = tris[rast.gridRepPoint[access] / 3 * 3];
        const float3 v2 = tris[rast.gridRepPoint[access] / 3 * 3 + 1];
        const float3 v3 = tris[rast.gridRepPoint[access] / 3 * 3 + 2];
        const float3 p = closest_point_on_triangle_to_point(v1, v2, v3, c0);
        // const float3 n0 = rast.gridPseudoNormal[access];
        #pragma unroll
        for (int s = 0; s < 26; s++)
        {
            const uint3 z1 = make_uint3(clamp(check[s], 0, N - 1));
            const float3 c1 = (i2f(z1) + 0.5f) / (float)N;
            const uint naccess = to_gidx(z1, N);
            const float3 nv1 = tris[rast.gridRepPoint[naccess] / 3 * 3];
            const float3 nv2 = tris[rast.gridRepPoint[naccess] / 3 * 3 + 1];
            const float3 nv3 = tris[rast.gridRepPoint[naccess] / 3 * 3 + 2];
            const float3 np = closest_point_on_triangle_to_point(nv1, nv2, nv3, c1);
            if (rast.gridDist[naccess] >= 0.86603f / N)
            {
                // if (!(dot(c0 - v0, n0) * dot(c1 - v0, n0) > 0))
                //     continue;
                if (!(dot(normalize(c0 - p), normalize(c1 - np)) > 0))
                    continue;
                cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }
            /*else
            {
                const float3 v1 = tris[rast.gridRepPoint[naccess]];
                const float3 n1 = rast.gridPseudoNormal[naccess];
                if (!(dot(c0 - v0, n0) * dot(c1 - v1, n1) > 0))
                    continue;
                cts_atomic_union(parents, shfm, shuffler(naccess, shfBitmask));
            }*/
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
