#include "kernel.h"

namespace diffRender
{
    __global__ void add_kernel(GArr2D<float> c, const GArr2D<float> a, const GArr2D<float> b)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int x = i / c.rows;
        int y = i % c.rows;
        c(x, y) = a(x, y) + b(x, y);
    }

    void launch_add(GArr2D<float> c, const GArr2D<float> a, const GArr2D<float> b)
    {
        cuExecute(c.cols * c.rows, add_kernel, c, a, b);
    }

} // namespace diffRender
