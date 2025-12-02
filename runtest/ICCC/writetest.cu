// writeincluster.cu
// 运行： ./writeincluster 16 5   # cluster_size=16, 每线程重复 5 次（可调）
// 第一个 SM (rank==0) 进行全局写，其他 SM 进行全局读

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
__device__ __forceinline__ float ld_ca_f32(  float* p) {
  float v;
#if __CUDA_ARCH__ >= 900
    //L1有固定8次miss，即使所有数据都是ld.cg。每次读取
  asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

// 强制经 L1 的全局写（.ca），用于数据地址 g_data[0]
__device__ __forceinline__ void st_ca_f32(  float* p, float v) {
#if __CUDA_ARCH__ >= 900
  //unsigned u = __float_as_uint(v);
  asm volatile("st.global.wb.f32 [%0], %1;" : : "l"(p), "f"(v) : "memory");
#else
  *p = v;
#endif
}

// 绕过 L1 的全局读（.cg），用于轮询控制变量 g_turn，避免污染 L1D 统计
__device__ __forceinline__ int ld_cg_s32(  int* p) {
  int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
#else
  v = *p;
#endif
  return v;
}

// smid有bug，模拟器src/gpu-cache.cc中的get-sid是正确的
__device__ __forceinline__ unsigned get_smid() {
  unsigned smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

// 注册需要维护cluster一致性的地址（由模拟器拦截处理）
// 这个函数在CUDA代码中提供空实现，实际处理由模拟器在instructions.cc中拦截
// 使用 __attribute__((noinline)) 防止函数被内联，确保能被模拟器拦截
// 添加 volatile 变量防止函数被完全优化掉
__device__ volatile int g_dummy_registration_flag = 0;

extern "C" __device__ __attribute__((noinline)) void gpgpusim_register_cluster_coherent_addr(void* addr) {
  // 添加副作用防止函数被优化掉：访问 volatile 变量
  // 实际处理由模拟器在call_impl中拦截
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1);
  //__threadfence();  // 添加内存屏障，确保函数不会被优化
  (void)addr;  // 避免未使用参数的警告
}


template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void sequential_write_read_kernel(  float* __restrict__ g_data,
                                  float* __restrict__ out,
                                  int /*span_unused*/, int repeat)
{
#if __CUDA_ARCH__ >= 900

  cg::cluster_group cluster = cg::this_cluster();
    int rank = cluster.block_rank();      // 0..CLUSTER_SIZE-1
    int tid  = threadIdx.x;

    //先都读，保证后续写是命中

    float* addr = &g_data[0];

    // 注册该地址为需要维护cluster一致性的地址
    // 只有注册的地址才会在写操作时设置cluster-state
    if (tid == 0 && rank == 0) {
      gpgpusim_register_cluster_coherent_addr(addr);
    }
    cluster.sync();  // 确保注册完成后再继续

    float acc = ld_ca_f32(addr);
    //float acc = 0.0f;
    printf("\nINITIAL [READ] rank=%d, read value=%f\n", rank, acc);

    cluster.sync();

for(int r = 0; r < repeat; r++){
  // 只用前 1 线程
  if (tid >= 1) return;

  // 串行化：只允许 rank==g_turn 的 CTA 开始；轮询用 .cg 读，避免走 L1
  if (tid == 0) {
    while (ld_cg_s32(&g_turn) != rank) { /* spin */ }
  }
  //__syncthreads();

  // 关键：rank==0 的 SM 进行全局写，其他 SM 进行读

  //float* addrst = &g_data[0];
  //float acc = 0.f;
  //acc = ld_ca_f32(addr);
  if (rank == 0) {
    // 第一个 SM：进行全局写
    float write_value = 10.0f + (float)r;  // 每次写不同的值以便测试
    st_ca_f32(addr, write_value);  // 通过 L1D 的全局写
    //acc = write_value;  // 记录写入的值
    if (tid == 0) {
        printf("\nCLUSTER-[WRITE] rank=%d, wrote value=%f\n", rank, write_value);
    }

  } else {
    // 其他 SM：进行全局读
    float read_value = ld_ca_f32(addr);  // 通过 L1D；若读共享生效，可从 peer L1 提供
    if (tid == 0) {
        printf("\nCLUSTER-[READ] rank=%d, read value=%f\n", rank, read_value);
    }
  }

  // 写回（保留原布局）：每 CTA 1 个标量
  out[rank * 1 + tid] = 1;

  // 交棒给下一个 rank；用 system fence 确保后继 .cg 读可见
  if (tid == 0) {
    atomicExch(&g_turn, (rank + 1) % CLUSTER_SIZE);
    printf("\nfinals of turn rank = %d\n", rank);
    if(rank == CLUSTER_SIZE-1)
        printf("\n本轮结束\n");
  }

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
  std::vector<float> h_data(1, 2.0f);
  float *d_data=nullptr, *d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_out, cluster_size * 1 * sizeof(float)));
  //CUDA_CHECK(cudaMemset(d_out, 0, cluster_size * 1 * sizeof(float)));

  int zero = 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_turn, &zero, sizeof(int)));

  dim3 grid(cluster_size, 1, 1);   // 一个簇：grid.x == CLUSTER_SIZE
  dim3 block(1, 1, 1);            // 1 线程/CTA（都读同一地址）

  //d_data[0]=520;
  switch (cluster_size) {
    case 2:
        sequential_write_read_kernel<2 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 4:
        sequential_write_read_kernel<4 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 8:
        sequential_write_read_kernel<8 ><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
    case 16:
        sequential_write_read_kernel<16><<<grid, block>>>(d_data, d_out, 0, repeat);
        break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());


  std::printf("ClusterSize=%d, Repeat=%d, SingleAddr, BlockSize=1\n", cluster_size, repeat);
  std::printf("First SM (rank=0) writes, other SMs read\n");
  //std::printf("Check: %s  (example out[0]=%.1f)\n", ok ? "OK" : "MISMATCH", h_out[0]);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
