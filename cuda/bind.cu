#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"

__global__ void custom_cuda_kernel(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor custom_cuda_func(const torch::Tensor &a, const torch::Tensor &b)
{
    auto c = torch::empty_like(a);
    int n = a.size(0);
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;
    custom_cuda_kernel<<<blocks, threads>>>((float* )a.data_ptr(), (float* )b.data_ptr(), (float* )c.data_ptr(), n);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_cuda_func",
          &custom_cuda_func,
          "example for pytorch cuda extension");
}
