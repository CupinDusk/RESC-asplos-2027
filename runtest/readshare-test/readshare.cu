#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cinttypes>
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

__global__ void sequential_cluster_read_kernel(const volatile float *input,
                                                float *output,
                                                unsigned long long *latencies,
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

  unsigned long long start = 0;
  unsigned long long stop = 0;
  float value = 0.0f;

  if (lane < data_count) {
    start = clock64();
    value = input[lane];
    stop = clock64();
    output[block * data_count + lane] = value;
    latencies[block * data_count + lane] = stop - start;
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
  unsigned long long *device_latencies = nullptr;

  CUDA_CHECK(cudaMalloc(&device_input, host_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, host_output.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_latencies,
                        host_output.size() * sizeof(unsigned long long)));

  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        host_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_output, 0,
                        host_output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemset(device_latencies, 0,
                        host_output.size() * sizeof(unsigned long long)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_next_sm_to_read, &zero, sizeof(int)));

  dim3 grid(kClusterSms);
  dim3 block(kDataPerSm);

  sequential_cluster_read_kernel<<<grid, block>>>(device_input, device_output,
                                                  device_latencies, kDataPerSm);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                        host_output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<unsigned long long> host_latencies(host_output.size(), 0);
  CUDA_CHECK(cudaMemcpy(host_latencies.data(), device_latencies,
                        host_latencies.size() * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));

  std::printf("Launched %d blocks (each %d threads) to model %d-SM cluster\n",
              grid.x, block.x, kClusterSms);
  for (int sm = 0; sm < kClusterSms; ++sm) {
    const int offset = sm * kDataPerSm;
    unsigned long long min_latency = host_latencies[offset];
    unsigned long long max_latency = host_latencies[offset];
    unsigned long long total_latency = 0;
    for (int lane = 0; lane < kDataPerSm; ++lane) {
      unsigned long long sample = host_latencies[offset + lane];
      if (sample < min_latency) {
        min_latency = sample;
      }
      if (sample > max_latency) {
        max_latency = sample;
      }
      total_latency += sample;
    }
    double average_latency =
        static_cast<double>(total_latency) / static_cast<double>(kDataPerSm);
    std::printf(
        "SM %d first element: %.1f | latency (cycles) min=%" PRIu64
        " avg=%.1f max=%" PRIu64 "\n",
        sm, host_output[offset], min_latency, average_latency, max_latency);
  }

  std::puts(
      "\nCompare the latency statistics between simulator builds. "
      "With cluster read sharing enabled, only the first SM should pay the "
      "full global memory latency while the remaining SMs see much lower "
      "values.");

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_latencies));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
