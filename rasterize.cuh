#pragma once
#include "commons.cuh"
#include "geometry.cuh"
#include "grid.cuh"


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
    const float * gridDist, int * outGridRepIdx
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

    // const float cmp = gridDist[access] + FLT_EPSILON;
    if (sqrt(point_to_tri_dist_sqr(v1, v2, v3, fxyz)) == gridDist[access])
    {
        // TODO: pseudo-normal for vertices? how to handle float-point errors?
        // https://dl.acm.org/doi/pdf/10.5555/2619648.2619655
        // Signed Distance Fields for Polygon Soup Meshes
        // https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B%C3%A6rentzen.pdf
        // Signed distance computation using the angle weighted pseudonormal
        // const float3 n = normalize(cross(v2 - v1, v3 - v1));
        // atomicAdd(&outGridPseudoNormal[access].x, n.x);
        // atomicAdd(&outGridPseudoNormal[access].y, n.y);
        // atomicAdd(&outGridPseudoNormal[access].z, n.z);
        // uint pt = tofs * 3;
        // if (sqrt(point_to_segment_dist_sqr(v2, v3, fxyz)) < cmp)
        //     pt = tofs * 3 + 1;
        // if (length(v3 - fxyz) < cmp)
        //     pt = tofs * 3 + 2;
        atomicMax(outGridRepIdx + access, tofs);
    }
}
