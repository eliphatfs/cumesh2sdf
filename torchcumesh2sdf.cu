#include "torchcumesh2sdf.h"
#include "main.cuh"

at::Tensor get_sdf(const at::Tensor tris, const int R, const float band, const int B)
{
    assert(tris.sizes()[1] == 3);
    assert(tris.sizes()[2] == 3);
    auto rast = rasterize_tris((float3*)tris.data_ptr<float>(), tris.sizes()[0], R, band, B, true);
    fill_signs((float3*)tris.data_ptr<float>(), R, rast, true);
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
    at::Tensor result = torch::from_blob(rast.gridDist, { R, R, R }, options).clone();
    return result;
}

void free_cached_memory()
{
    clear_raster_alloc_cache();
    clear_sign_alloc_cache();
}
