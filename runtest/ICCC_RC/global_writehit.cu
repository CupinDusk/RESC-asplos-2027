// writeincluster_release_acquire.cu
// 运行： ./writeincluster 2 5   # cluster_size=2, repeat=5
// rank==0: producer, rank==1: consumer

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <cuda/atomic>   // <- NEW: cuda::atomic_ref
namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do { auto e=(x); if(e!=cudaSuccess){                     \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,              \
               cudaGetErrorString(e)); std::exit(1);} } while(0)

__device__ int g_turn;  // 0: producer may produce (data not ready), 1: data ready for consumer

// ---------------- PTX helpers ----------------

// 强制经 L1 的全局读（.ca）——保留但本版本不用于“正确性版数据路径”
__device__ __forceinline__ float ld_ca_f32(float* p) {
  float v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

// 强制经 L1 的全局写（.wb）——保留但本版本不用于“正确性版数据路径”
__device__ __forceinline__ void st_wb_f32(float* p, float v) {
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.wb.f32 [%0], %1;" : : "l"(p), "f"(v) : "memory");
#else
  *p = v;
#endif
}

// 绕过 L1 的全局读（.cg）——用于 flag/poll，也用于数据（正确性）
__device__ __forceinline__ int ld_cg_s32(int* p) {
  int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

__device__ __forceinline__ float ld_cg_f32(float* p) {
  float v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

__device__ __forceinline__ void st_cg_f32(float* p, float v) {
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(p), "f"(v) : "memory");
#else
  *p = v;
#endif
}

// smid
__device__ __forceinline__ unsigned get_smid() {
  unsigned smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

// ---------------- (optional) simulator hook keep as-is ----------------
__device__ volatile int g_dummy_registration_flag = 0;

extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr) {
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1);
  (void)addr;
}

// ---------------- Kernel ----------------
template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void producer_consumer_kernel(float* __restrict__ g_data,
                              float* __restrict__ out,
                              int /*span_unused*/, int repeat)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();   // 0..CLUSTER_SIZE-1
  printf("rank=%d\n", rank);
  int tid  = threadIdx.x;

  if (tid >= 1) return;             // 保持你原先“单线程 CTA”框架

  float* addr = &g_data[0];
  float initial_value = ld_ca_f32(addr);
  printf("\nINITIAL [READ] rank=%d, initial value=%f\n", rank, initial_value);

  // (可选) 保留你的注册逻辑
  if (tid == 0 && rank == 0) {
    gpgpusim_register_cluster_coherent_addr(addr);
  }
  cluster.sync();

  // device-scope atomic flag (works across SMs)
  cuda::atomic_ref<int, cuda::thread_scope_device> flag(g_turn);

  // 约束：这里按你的要求固定 cluster_size=2
  // rank0: producer, rank1: consumer
  for (int r = 0; r < repeat; r++) {

    if (rank == 0) {
      // -------- Producer --------
      // 等待 consumer ack：flag==0
      //while (flag.load(cuda::std::memory_order_acquire) != 0) { /* spin */ }

      float write_value = 10.0f + (float)r;

      // 正确性：绕过 L1 写入，使数据到达 L2（一致性点）
      st_cg_f32(addr, write_value);
      //float read_value2 = ld_ca_f32(addr);
      // 发布：让 consumer 在看到 flag==1 时必然看到 write_value
      flag.store(1, cuda::std::memory_order_release);

      if (tid == 0) {
        printf("\nCLUSTER-[WRITE] rank=%d, wrote value=%f\n", rank, write_value);
      }

    } else { // rank == 1
      // -------- Consumer --------
      // 等待 producer 发布：flag==1
      while (flag.load(cuda::std::memory_order_acquire) != 1) { /* spin */ }

      // 正确性：绕过 L1 读取（否则可能命中自己的旧 L1 行）
      float read_value = ld_cg_f32(addr);

      if (tid == 0) {
        printf("\nCLUSTER-[READ] rank=%d, read value=%f\n", rank, read_value);
      }

      // ack：告诉 producer 本轮已消费完，可以进入下一轮
      //flag.store(0, cuda::std::memory_order_release);
    }

    out[rank] = 1.0f; // 保留原布局/框架

    if(rank == CLUSTER_SIZE-1){
      flag.store(0, cuda::std::memory_order_release);
      printf("\n本轮结束\n");
    }

    cluster.sync();
  }


#endif
}

// ---------------- Host ----------------
int main(int argc, char** argv) {
  int cluster_size = 2;
  int repeat       = 5;
  if (argc >= 2) cluster_size = std::atoi(argv[1]);
  if (argc >= 3) repeat       = std::atoi(argv[2]);



  std::vector<float> h_data(1, 2.0f);
  float *d_data=nullptr, *d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, cluster_size * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_turn, &zero, sizeof(int)));

  dim3 grid(cluster_size, 1, 1);
  dim3 block(1, 1, 1);

  //producer_consumer_kernel<2><<<grid, block>>>(d_data, d_out, 0, repeat);
  switch (cluster_size) {
    case 2:
        producer_consumer_kernel<2 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 4:
        producer_consumer_kernel<4 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 8:
        producer_consumer_kernel<8 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 16:
        producer_consumer_kernel<16><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("\nClusterSize=%d, Repeat=%d, SingleAddr, BlockSize=1\n", cluster_size, repeat);
  std::printf("rank0=producer (release store), rank1=consumer (acquire load)\n");

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
