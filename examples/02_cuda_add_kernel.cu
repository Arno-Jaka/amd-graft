#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * 2.0f;
    }
}

__global__ void reduce_kernel(const float* in, float* out, int n) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    scratch[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            total += scratch[i];
        }
        out[0] = total;
    }
}

int main() {
    const int n = 1 << 20;
    float* device_a = nullptr;
    float* device_b = nullptr;

    if (cudaMalloc(&device_a, n * sizeof(float)) != cudaSuccess) {
        return 1;
    }
    cudaMalloc(&device_b, n * sizeof(float));

    cudaMemcpy(device_a, device_b, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((n + 255) / 256);
    dim3 threads(256);
    add_kernel<<<blocks, threads>>>(device_a, device_b, n);
    cudaDeviceSynchronize();

    float* host = new float[n];
    cudaMemcpy(host, device_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    delete[] host;
    return 0;
}