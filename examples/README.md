Examples go here.

Priority: get two or three files that are genuinely hard but tractable
converting, verifying, and benchmarking cleanly before building anything
else. The demo depends on these, not on agent behaviour.

01_softmax.cu            custom softmax kernel, classic warp-assumption trap
02_gemm_extension.py     CUDA extension module, exercises hipify path
03_torch_cuda_pipeline.py  torch.cuda.* call surface in idiomatic PyTorch