#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

at::Tensor get_sdf(const at::Tensor tris, const int R, const float band, const int B);
void free_cached_memory();
