#include <torch/extension.h>

#include "torchcumesh2sdf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_sdf", &get_sdf, "get_sdf (CUDA)");
}
