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

constexpr int kClusterSms = 8;
constexpr int kDataPerSm = 32;

__device__ int g_next_sm_to_read = 0;

__global__ void sequential_cluster_read_kernel(const float *input,
                                                float *output,
                                                int data_count) {
  int block = blockIdx.x;
  int lane = threadIdx.x;

  if (lane == 0) {
    // Ensure only one block at a time proceeds to read the shared data.
    while (atomicAdd(&g_next_sm_to_read, 0) != block) {
    }
    __threadfence_block();
  }
  __syncthreads();

  if (lane < data_count) {
    float value = input[lane];
    output[block * data_count + lane] = value;
  }
  __syncthreads();

  if (lane == 0) {
    __threadfence();
    atomicAdd(&g_next_sm_to_read, 1);
  }
}

int main() {
  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  if (props.multiProcessorCount < kClusterSms) {
    std::fprintf(stderr,
                 "Device has %d SMs, but this test requires at least %d SMs.\n",
                 props.multiProcessorCount, kClusterSms);
    return EXIT_FAILURE;
  }

  std::vector<float> host_input(kDataPerSm, 1.0f);
  std::vector<float> host_output(kClusterSms * kDataPerSm, 0.0f);

  float *device_input = nullptr;
  float *device_output = nullptr;

  CUDA_CHECK(cudaMalloc(&device_input, host_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, host_output.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        host_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_output, 0,
                        host_output.size() * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_next_sm_to_read, &zero, sizeof(int)));

  dim3 grid(kClusterSms);
  dim3 block(kDataPerSm);

  sequential_cluster_read_kernel<<<grid, block>>>(device_input, device_output,
                                                  kDataPerSm);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                        host_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("Launched %d blocks (each %d threads) to model %d-SM cluster\n",
              grid.x, block.x, kClusterSms);
  for (int sm = 0; sm < kClusterSms; ++sm) {
    std::printf("SM %d first element: %.1f\n", sm,
                host_output[sm * kDataPerSm]);
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
