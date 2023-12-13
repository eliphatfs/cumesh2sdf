#include <cuda.h>
#include <stdio.h>
#include "helper_math.h"

constexpr const float EPS = 1e-10f;

__forceinline__ __device__ float lensqr(float3 v)
{
    return dot(v, v);
}

__forceinline__ __device__ float point_to_segment_dist_sqr(float3 v, float3 w, float3 p)
{
  // Return minimum distance between line segment vw and point p
  w -= v;
  p -= v;
  const float l2 = lensqr(w);  // i.e. |w-v|^2 -  avoid a sqrt
  if (l2 < EPS) return lensqr(p);   // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line. 
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  // We clamp t from [0,1] to handle points outside the segment vw.
  const float t = clamp(dot(p, w) / l2, 0.0f, 1.0f);
  const float3 projection = t * w;  // Projection falls on the segment
  return lensqr(p - projection);
}

__forceinline__ __device__ float point_to_tri_dist_sqr(float3 v1, float3 v2, float3 v3, float3 p)
{
    const float d1 = point_to_segment_dist_sqr(v1, v2, p);
    const float d2 = point_to_segment_dist_sqr(v2, v3, p);
    const float d3 = point_to_segment_dist_sqr(v3, v1, p);

    const float min_edge = fminf(fminf(d1, d2), d3);

    const float3 e0 = v2 - v1;
    const float3 e1 = v3 - v1;

    const float3 noru = cross(e0, e1);
    const float scl = length(noru);
    if (scl < EPS) return min_edge;  // 0-area tri
    const float3 nor = noru / scl;

    const float3 proj = p - (dot(p, nor) - dot(v1, nor)) * nor;
    const float3 e2 = proj - v1;

    const float dot00 = dot(e0, e0);
    const float dot01 = dot(e0, e1);
    const float dot11 = dot(e1, e1);
    const float dot02 = dot(e0, e2);
    const float dot12 = dot(e1, e2);

    const float denom = dot00 * dot11 - dot01 * dot01;
    // TODO: denom ~ 0 when cosc ~ 1. do we need another degenerate check?
    if (denom < EPS) return min_edge;
    const float invDenom = 1.0 / denom;
    const float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    const float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    const float uc = clamp(u, 0.0f, 1.0f);
    const float vc = clamp(v, 0.0f, 1.0f - uc);

    const float3 prc = v1 + uc * e0 + vc * e1;
    // printf("%.3f %.3f; %.3f %.3f %.3f; %.3f %.3f %.3f\n", uc, vc, proj.x, proj.y, proj.z, prc.x, prc.y, prc.z);
    // printf("%.3f %.3f\n", lensqr(p - prc), min_edge);
    return fminf(lensqr(p - prc), min_edge);
}
