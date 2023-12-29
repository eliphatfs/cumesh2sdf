#include "torchcumesh2sdf.h"
#include "main.cuh"

at::Tensor get_sdf(const at::Tensor tris, const int R, const float band)
{
    assert(tris.sizes()[1] == 3);
    assert(tris.sizes()[2] == 3);
    auto rast = rasterize_tris((float3*)tris.data_ptr<float>(), tris.sizes()[0], R, band);
    // fill_signs((float3*)tris.data_ptr<float>(), R, rast);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    at::Tensor result = torch::from_blob(rast.gridDist, { R, R, R }, options).clone();
    rast.free();
    return result;
}
