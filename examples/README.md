# Examples

Each example is a module satisfying the entrypoint contract:

| Element | Purpose |
| --- | --- |
| `solve(x_cpu) -> tensor` | entry point, shape-generic, CPU tensor in/out |
| `kernel(x) -> tensor` | device-resident, no transfers, safe to benchmark |
| `OPERATION` | key into the reference registry in `graft/core/reference.py` |

`solve` must accept any shape: `verify/cases.py` drives it across a range
of inputs, and a kernel hardcoded to one tensor size cannot be verified.

`solve` and `kernel` are separate on purpose. `solve` moves data to the
device and back, so timing it measures transfer overhead plus kernel time.
`kernel` operates on a tensor already resident on the device, which is
what `bench/` needs in order to compare a port against the reference on
equal footing.

## Available

| File | Exercises |
| --- | --- |
| `01_torch_cuda_softmax.py` | `torch.cuda.*` call surface, device resolution, hand-rolled softmax reduction |

## Planned

| File | Exercises |
| --- | --- |
| `02_hardcoded_device.py` | literal device indices, so the rewriter produces a visible diff |
| `03_gemm_extension.py` | custom CUDA extension, the path that genuinely needs HIPIFY |

`verify_example.py` runs example 01 end to end: converts it, then checks
`solve` against `reference` across six shapes. It runs in CI, so a
converter regression fails the build rather than the live demo.