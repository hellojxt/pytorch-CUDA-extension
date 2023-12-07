import torch
from torch.utils.cpp_extension import load
from glob import glob
import os

os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join("cuda", "build")
Debug = False # compile with debug flag
verbose = True # show compile command
cuda_files = glob("cuda/*.cu") # source files
include_dirs = ["cuda/include"] # include directories
cflags = "--extended-lambda --expt-relaxed-constexpr " # nvcc flags
if Debug:
    cflags += "-G -g -O0"
else:
    cflags += "-O3"
cuda_module = load(
    name="cuda_module",
    sources=cuda_files,
    extra_include_paths=include_dirs,  
    extra_cflags=[cflags],
    verbose=verbose,
)

N = 10000
a = torch.arange(N, device="cuda", dtype=torch.float32)
b = torch.arange(N, device="cuda", dtype=torch.float32)

c = cuda_module.custom_cuda_func(a, b)

assert torch.allclose(c, a + b)

