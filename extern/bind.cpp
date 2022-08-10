#include <torch/extension.h>
#include "kernel.h"
#include "array3D.h"

using namespace diffRender;

void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b) {
    GArr2D<float> a_arr((float *)a.data_ptr(), a.strides()[0], a.strides()[1]);
    GArr2D<float> b_arr((float *)b.data_ptr(), b.strides()[0], b.strides()[1]);
    GArr2D<float> c_arr((float *)c.data_ptr(), c.strides()[0], c.strides()[1]);
    launch_add(c_arr, a_arr, b_arr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}
