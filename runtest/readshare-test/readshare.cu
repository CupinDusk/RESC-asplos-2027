#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(stmt)                                                      \
  do {                                                                        \
    cudaError_t err = stmt;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "Failed to run %s (%s) at %s:%d\n", #stmt,         \
                   cudaGetErrorString(err), __FILE__, __LINE__);              \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// === 工作量缩放：把任意整数缩放到原来的 5%（向上取整，至少为 1） ===
#ifndef WORKLOAD_SCALE_NUM
#define WORKLOAD_SCALE_NUM 5
#endif
#ifndef WORKLOAD_SCALE_DEN
#define WORKLOAD_SCALE_DEN 100
#endif
#define SCALE_5P(x) ( ((int64_t)(x) * WORKLOAD_SCALE_NUM + WORKLOAD_SCALE_DEN - 1) / WORKLOAD_SCALE_DEN > 0 ? \
                      (int)(((int64_t)(x) * WORKLOAD_SCALE_NUM + WORKLOAD_SCALE_DEN - 1) / WORKLOAD_SCALE_DEN) : 1 )

__device__ int g_cluster_ready_flag;

__global__ void cluster_l1_broadcast_kernel(const float *input, float *output,
                                            int iterations, int span) {
  if (blockIdx.x == 0) {
    float accum = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
      accum += input[iter % span];
    }
    if (threadIdx.x == 0) {
      output[0] = accum;
      __threadfence();
      atomicExch(&g_cluster_ready_flag, 1);
    }
  } else {
    if (threadIdx.x == 0) {
      while (atomicAdd(&g_cluster_ready_flag, 0) == 0) {
      }
      __threadfence();
    }
    __syncthreads();

    float accum = 0.0f;
    for (int iter = 0; iter < iterations; ++iter) {
      accum += input[iter % span];
    }

    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    output[out_index] = accum;
  }
}

int main() {
  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  // 原始规模
  const int span = 32;
  const int iterations_orig = 2048;
  // 缩到 5%
  const int iterations_scaled = SCALE_5P(iterations_orig);

  const int threads_per_block = 64;
  int blocks = props.multiProcessorCount;
  if (blocks < 2) {
    blocks = 2;  // 需要至少两个 block 才能体现跨 SM
  }

  std::vector<float> host_input(span, 1.0f);
  std::vector<float> host_output(blocks * threads_per_block, 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, span * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output,
                        host_output.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        span * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_output, 0,
                        host_output.size() * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_cluster_ready_flag, &zero, sizeof(int)));

  dim3 grid(blocks);
  dim3 block(threads_per_block);
  cluster_l1_broadcast_kernel<<<grid, block>>>(device_input, device_output,
                                               iterations_scaled, span);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                        host_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("Launched %d blocks (%d threads each) on %d SMs\n", blocks,
              threads_per_block, props.multiProcessorCount);
  std::printf("Scaled iterations: %d (from %d)\n", iterations_scaled, iterations_orig);
  std::printf("First element after kernel: %.1f\n", host_output[0]);

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
