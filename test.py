import time
import numpy as np
import torch
from torch.utils.cpp_extension import load
import os

os.environ['TORCH_EXTENSIONS_DIR'] = 'extern/'
cuda_module = load(name="diffRender",
                   sources=["extern/bind.cpp", "extern/kernel.cu"],
                   extra_include_paths=["extern/include"],
                   extra_cflags=['-O3'],
                  #  verbose=True,
                   )

n = 2
a = torch.arange(n*n).reshape(n, n).cuda().float()
b = torch.arange(n*n).reshape(n, n).cuda().float()
c = torch.zeros(n, n).cuda()
cuda_module.my_torch_func(c, a, b)
print(a)
print(b)
print(c)