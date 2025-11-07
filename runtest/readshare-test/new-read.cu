// readshare.cu
// 运行： ./readshare 16 5   # cluster_size=16, 每线程重复读 5 次（可调）

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#define TUNM 64

#define CUDA_CHECK(x) do { auto e=(x); if(e!=cudaSuccess){                     \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,              \
               cudaGetErrorString(e)); std::exit(1);} } while(0)

__device__ int g_turn;  // 簇内当前允许执行的 rank：0..CLUSTER_SIZE-1

// 强制经 L1 的全局读（.ca），用于数据地址 g_data[0]
__device__ __forceinline__ float ld_ca_f32(const float* p) {
  float v;
#if __CUDA_ARCH__ >= 900  
    //L1有固定8次miss，即使所有数据都是ld.cg。每次读取
  asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

// 绕过 L1 的全局读（.cg），用于轮询控制变量 g_turn，避免污染 L1D 统计
__device__ __forceinline__ int ld_cg_s32(const int* p) {
  int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

__device__ __forceinline__ unsigned get_smid() {
  unsigned smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}


template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void sequential_readshare_kernel(const float* __restrict__ g_data,
                                 float* __restrict__ out,
                                 int /*span_unused*/, int repeat)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();      // 0..CLUSTER_SIZE-1
  const int tid  = threadIdx.x;

  // 只用前 8 线程；每线程都读同一地址 g_data[0]
  if (tid >= 8) return;

  // 串行化：只允许 rank==g_turn 的 CTA 开始；轮询用 .cg 读，避免走 L1
  if (tid == 0) {
    while (ld_cg_s32(&g_turn) != rank) { /* spin */ }
    __threadfence();   // 线程序（块内）
  }
  __syncthreads();

  // ——关键：所有线程都读同一个地址（同一 cache line）——
  const float* addr = &g_data[0];
  float acc = 0.f;
  #pragma unroll 1
  for (int r = 0; r < repeat; ++r) {
    acc += ld_ca_f32(addr);   // 通过 L1D；若读共享生效，后续 SM 可从 peer L1 提供
    if(tid == 0)
        printf("rank=%d, smid=%u, tid=%d, addr =%f", rank, get_smid(), tid, *addr);
  }

  // 写回（保留原布局）：每 CTA 8 个标量
  out[rank * 8 + tid] = acc;

  // 交棒给下一个 rank；用 system fence 确保后继 .cg 读可见
  if (tid == 0) {
    atomicExch(&g_turn, (rank + 1) % CLUSTER_SIZE);
    __threadfence_system();
  }
#endif
}

// ---------------- Host ----------------
int main(int argc, char** argv) {
  int cluster_size = 16;
  int repeat       = 5;     // 每线程重复读次数
  if (argc >= 2) cluster_size = std::atoi(argv[1]);
  if (argc >= 3) repeat       = std::atoi(argv[2]);

  if (!(cluster_size==2 || cluster_size==4 || cluster_size==8 || cluster_size==16)) {
    std::printf("cluster_size must be 2/4/8/16\n"); return 1;
  }

  // 仅 1 个元素：所有 CTA/线程都读同一地址
  std::vector<float> h_data(1, 1.0f);
  float *d_data=nullptr, *d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, cluster_size * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, cluster_size * 8 * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_turn, &zero, sizeof(int)));

  dim3 grid(cluster_size, 1, 1);   // 一个簇：grid.x == CLUSTER_SIZE
  dim3 block(8, 1, 1);            // 8 线程/CTA（都读同一地址）

  switch (cluster_size) {
    case 2:  sequential_readshare_kernel<2 ><<<grid, block>>>(d_data, d_out, 0, repeat); break;
    case 4:  sequential_readshare_kernel<4 ><<<grid, block>>>(d_data, d_out, 0, repeat); break;
    case 8:  sequential_readshare_kernel<8 ><<<grid, block>>>(d_data, d_out, 0, repeat); break;
    case 16: sequential_readshare_kernel<16><<<grid, block>>>(d_data, d_out, 0, repeat); break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(cluster_size * 8);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));

  // 期望每个元素都是 repeat * 1.0f
  bool ok = true;
  for (int r = 0; r < cluster_size && ok; ++r) {
    for (int i = 0; i < 8; ++i) {
      float expect = float(repeat);
      if (fabsf(h_out[r*8 + i] - expect) > 1e-3f * expect) { ok = false; break; }
    }
  }
  std::printf("ClusterSize=%d, Repeat=%d, SingleAddr, BlockSize=8\n", cluster_size, repeat);
  std::printf("Check: %s  (example out[0]=%.1f)\n", ok ? "OK" : "MISMATCH", h_out[0]);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
