"""Prove Graft's ported HIP actually computes the right answer on real hardware.

Run inside the ROCm container on the MI300X droplet.

This is deliberately not a self-contained test. It calls Graft to produce the
ported source, splices a verifying `main` around the kernels Graft emitted,
compiles that with hipcc for the real gfx942 target, and runs it. So the
thing under test is the translator's output, not a hand-written HIP file that
merely resembles it.

Usage:  PYTHONPATH=src python3 tools/verify_hip.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from graft.core import converter, coverage  # noqa: E402

SOURCE = REPO / "examples" / "02_cuda_add_kernel.cu"
ARCH = "gfx942"

# Spliced in place of the example's stub main, which launches the kernel and
# returns 0 without ever looking at the output.
HARNESS = r"""
// --- graft verification harness -----------------------------------------
#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;

static void check(const char* what, float worst, float tol) {
    bool ok = worst <= tol;
    if (!ok) ++failures;
    printf("  %-28s max_abs_error=%.3e  tol=%.0e  %s\n",
           what, worst, tol, ok ? "PASS" : "FAIL");
}

int main() {
    const int n = 1 << 16;
    float *da = nullptr, *db = nullptr;
    hipMalloc(&da, n * sizeof(float));
    hipMalloc(&db, n * sizeof(float));

    std::vector<float> host_a(n), host_b(n, 0.0f), ref(n);
    for (int i = 0; i < n; ++i) {
        host_a[i] = (float)((i * 37 % 101) - 50) / 7.0f;
        ref[i] = host_a[i] * 2.0f;  // what add_kernel must produce
    }
    hipMemcpy(da, host_a.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(db, 0, n * sizeof(float));

    // --- kernel 1: add_kernel, elementwise, n not a multiple of blockDim
    add_kernel<<<(n + 255) / 256, 256>>>(da, db, n);
    hipDeviceSynchronize();
    hipMemcpy(host_b.data(), db, n * sizeof(float), hipMemcpyDeviceToHost);
    float worst = 0.0f;
    for (int i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(host_b[i] - ref[i]));
    printf("add_kernel (n=%d, tail=%d)\n", n, n % 256);
    check("elementwise scale by 2", worst, 1e-6f);

    // --- kernel 2: reduce_kernel, block reduction through shared memory.
    // Also exercises __shared__ plus __syncthreads across a 64-wide
    // wavefront, which is the construct most likely to be silently wrong
    // after a CUDA-to-HIP rename.
    const int block = 256;
    hipMemset(db, 0, n * sizeof(float));
    float *d_sum = db;
    reduce_kernel<<<1, block>>>(da, d_sum, block);
    hipDeviceSynchronize();
    float got = 0.0f;
    hipMemcpy(&got, d_sum, sizeof(float), hipMemcpyDeviceToHost);
    double expect = 0.0;
    for (int i = 0; i < block; ++i) expect += host_a[i];
    double err = std::fabs((double)got - expect);
    printf("reduce_kernel (block=%d, wavefront=64)\n", block);
    check("shared-memory block sum", (float)err, 1e-3);

    hipFree(da);
    hipFree(db);
    printf("\n%s\n", failures ? "HIP VERIFICATION FAILED" : "HIP VERIFICATION PASSED");
    return failures ? 1 : 0;
}
"""


def strip_main(source: str) -> str:
    """Remove the example's stub main so the harness can replace it."""
    index = source.find("int main(")
    if index == -1:
        raise SystemExit("ported source has no main(); cannot splice a harness")
    return source[:index]


def main() -> int:
    result = converter.convert(
        SOURCE.read_text(encoding="utf-8"), SOURCE
    )
    ported = result.ported_source

    print("=" * 72)
    print("Graft conversion")
    print("=" * 72)
    print(f"backend  : {result.backend}")
    print(f"changes  : {len(result.changes)}")
    for change in result.changes[:10]:
        print(f"  L{change['line']:<4} {change['symbol']:<26} {change['reason']}")

    assessment = coverage.assess(ported)
    print(f"verdict  : {assessment.verdict}")
    print(f"findings : {len(assessment.findings)}")
    for finding in assessment.findings[:10]:
        print(f"  [{finding.severity}] {finding.symbol}: {finding.reason}")

    if result.backend != "hipify-perl":
        print(
            "\nWARNING: hipify-perl was not used, so this verified the table "
            "fallback rather than the official translation."
        )

    harnessed = strip_main(ported) + HARNESS
    workdir = Path(tempfile.mkdtemp(prefix="graft-hip-verify-"))
    hip_file = workdir / "verify.hip"
    hip_file.write_text(harnessed, encoding="utf-8")
    binary = workdir / "verify"

    print("\n" + "=" * 72)
    print(f"hipcc --offload-arch={ARCH}")
    print("=" * 72)
    compile_run = subprocess.run(
        [
            "hipcc",
            f"--offload-arch={ARCH}",
            "-O2",
            "-Wno-unused-result",
            "-o",
            str(binary),
            str(hip_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if compile_run.returncode != 0:
        print(compile_run.stdout[-2000:])
        print(compile_run.stderr[-3000:])
        print("\nCOMPILE FAILED")
        return 1
    print("compiled cleanly")

    print("\n" + "=" * 72)
    print("execution")
    print("=" * 72)
    # The ROCm pip wheel ships its runtime libs in the package tree rather
    # than on the default loader path, so the binary cannot find them
    # without this. Located here instead of hardcoded so the script works
    # on any ROCm install.
    env = dict(os.environ)
    sdk_lib = next(
        (
            str(path.parent)
            for path in sorted(REPO.glob("**/_rocm_sdk_core/lib/libamdhip64.so*"))
        ),
        None,
    )
    if sdk_lib is None:
        # Fall back to the interpreter's site-packages.
        import sysconfig

        candidate = Path(sysconfig.get_paths()["purelib"]) / "_rocm_sdk_core" / "lib"
        if candidate.is_dir():
            sdk_lib = str(candidate)
    if sdk_lib:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [sdk_lib, env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)

    execution = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=300, env=env
    )
    print(execution.stdout.strip() or "(no output)")
    if execution.stderr.strip():
        print("stderr:", execution.stderr.strip()[:1000])
    return execution.returncode


if __name__ == "__main__":
    raise SystemExit(main())
