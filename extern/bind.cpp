#include <torch/extension.h>
#include "kernel.h"

using namespace diffRender;

void my_torch_func(torch::Tensor &c,
                    const torch::Tensor &a,
                    const torch::Tensor &b)
{
    GArr2D<float> a_arr((float *)a.data_ptr(), a.strides()[0], a.strides()[1]);
    GArr2D<float> b_arr((float *)b.data_ptr(), b.strides()[0], b.strides()[1]);
    GArr2D<float> c_arr((float *)c.data_ptr(), c.strides()[0], c.strides()[1]);
    launch_add(c_arr, a_arr, b_arr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_torch_func",
          &my_torch_func,
          "matrix add kernel warpper");
}
