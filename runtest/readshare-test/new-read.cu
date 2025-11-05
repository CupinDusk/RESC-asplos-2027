// readshare.cu
// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo readshare.cu -o readshare -lcudart
// 运行示例： ./readshare 16 50000   # 16 个 CTA（簇大小 16），每 CTA 读同一组 64 地址，重复 5 万次

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#define CUDA_CHECK(stmt)                                                      \
  do {                                                                        \
    cudaError_t err = (stmt);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "Failed to run %s (%s) at %s:%d\n", #stmt,         \
                   cudaGetErrorString(err), __FILE__, __LINE__);              \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

__device__ int g_turn; // 全局“轮到谁”指示（0..cluster_size-1）

// 让 16 个 CTA（尽量 16 个 SM）依次读同一组 64 地址；
// 代码里都是普通的 global load（通过 volatile 避免被编译器优化），
// 你的模拟器可在后续 CTA 的读上触发“从前一个 SM 的 L1 读共享”而非再次走 L2/DRAM。
template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void sequential_readshare_kernel(const float* __restrict__ g_data,
                                 float* __restrict__ out,
                                 int span, int repeat)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();      // 0..CLUSTER_SIZE-1
  const int tid  = threadIdx.x;               // 我们用 64 线程/块，每线程读 1 个地址
  const int lane = tid & 63;

  // 只读 64 个元素（每线程一个），剩余线程闲置
  if (tid >= 64) return;

  // 顺序执行：rank==g_turn 的 CTA 才能开始读取；其余 CTA 自旋等待
  if (tid == 0) {
    // busy-wait 等到轮到我
    while (atomicAdd(&g_turn, 0) != rank) { /* spin */ }
    __threadfence(); // 保证后续读不会被重排到等待之前
  }
  __syncthreads();

  // 重复读取：同一地址重复 read 'repeat' 次（普通 global load）
  volatile const float* vptr = (volatile const float*)g_data; // 防止被合并/提升
  float acc = 0.f;
  const int idx = lane % span; // 0..63
  #pragma unroll 1
  for (int r = 0; r < repeat; ++r) {
    float x = vptr[idx];       // global load（由模拟器决定是否命中“读共享”）
    acc += x;
  }

  // 写回输出：每 CTA 64 个结果
  out[rank * 64 + lane] = acc;
  __threadfence();

  // 轮到下一个 CTA
  if (tid == 0) {
    atomicExch(&g_turn, (rank + 1) % CLUSTER_SIZE);
  }
#endif
}

// ---------------- Host ----------------
int main(int argc, char** argv) {
  int cluster_size = 16;
  int repeat       = 5;   // 默认 5 万次（≈1 分钟内结束；可按机器调整）

  if (argc >= 2) cluster_size = std::atoi(argv[1]);
  if (argc >= 3) repeat       = std::atoi(argv[2]);
  if (!(cluster_size==2 || cluster_size==4 || cluster_size==8 || cluster_size==16)) {
    std::printf("cluster_size must be 2/4/8/16\n"); return 1;
  }

  // 64 个只读元素（所有 CTA 都读同一组地址）
  const int span = 64;
  std::vector<float> h_data(span, 1.0f);
  float *d_data = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, span * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), span * sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, cluster_size * 64 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, cluster_size * 64 * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_turn, &zero, sizeof(int)));

  dim3 grid(cluster_size, 1, 1);
  dim3 block(64, 1, 1); // 恰好 64 线程/块，每线程一个地址

  // 为不同 cluster_size 实例化（避免运行时分支）
  switch (cluster_size) {
    case 2:  sequential_readshare_kernel<2 ><<<grid, block>>>(d_data, d_out, span, repeat); break;
    case 4:  sequential_readshare_kernel<4 ><<<grid, block>>>(d_data, d_out, span, repeat); break;
    case 8:  sequential_readshare_kernel<8 ><<<grid, block>>>(d_data, d_out, span, repeat); break;
    case 16: sequential_readshare_kernel<16><<<grid, block>>>(d_data, d_out, span, repeat); break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(cluster_size * 64);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // 简单校验：每个输出都应是 repeat * 1.0f
  bool ok = true;
  for (int r = 0; r < cluster_size && ok; ++r) {
    for (int i = 0; i < 64; ++i) {
      float expect = repeat * 1.0f;
      float got = h_out[r*64 + i];
      if (fabsf(got - expect) > 1e-3f * expect) { ok = false; break; }
    }
  }

  printf("ClusterSize=%d, Repeat=%d, Span=64, BlockSize=64\n", cluster_size, repeat);
  printf("Check: %s  (example out[0]=%.1f)\n", ok ? "OK" : "MISMATCH", h_out[0]);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
