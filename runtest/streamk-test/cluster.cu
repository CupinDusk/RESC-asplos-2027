// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo clustersim_streamk_dsm_gemm.cu -o clustersim_streamk_dsm_gemm
// 运行（ClusterSim）：
//   1) 按 README 构建并 source enable_simulator.sh
//   2) ./clustersim_streamk_dsm_gemm
//
// 亦可用 cudaLaunchKernelExC 启动（见 main() 中 USE_EX_LAUNCH 开关）

#include <cstdio>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ======== 可调参数（保持整除以简化逻辑）========
#ifndef M
#define M 64
#endif
#ifndef N
#define N 64
#endif
#ifndef K
#define K 64
#endif

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 32   // K 被二分：32 + 32
#endif

static_assert(M % TILE_M == 0 && N % TILE_N == 0 && K % TILE_K == 0, "dims must be multiples");

// 线性索引
__host__ __device__ inline int idx2(int r, int c, int ld) { return r*ld + c; }

// ======== Kernel：2-CTA Cluster 的极简 Stream-K×DSM GEMM ========
__global__ __cluster_dims__(2,1,1)
void streamk_cluster_dsm_gemm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int ldA, int ldB, int ldC)
{
#if __CUDA_ARCH__ >= 900
  // 本微基准：一个 cluster (=2 CTAs) 只负责 C 的 [0:TILE_M)×[0:TILE_N) 这个 tile
  constexpr int row0 = 0;
  constexpr int col0 = 0;

  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();   // 0 或 1
  int peer = rank ^ 1;
  const int tid = threadIdx.x;

  // 动态共享内存：A_tile + B_tile + C_partial
  extern __shared__ float smem[];
  float* A_s = smem;                                           // TILE_M × TILE_K
  float* B_s = A_s + TILE_M * TILE_K;                          // TILE_K × TILE_N
  float* C_s = B_s + TILE_K * TILE_N;                          // TILE_M × TILE_N

  // 对端 CTA 的 DSM 指针（将本 CTA 的 SMEM 地址映射为“对端 SMEM 地址”）
  float* A_peer = cluster.map_shared_rank(A_s, peer);
  float* B_peer = cluster.map_shared_rank(B_s, peer);
  float* C_peer = cluster.map_shared_rank(C_s, peer);

  // 清零本 CTA 的 C 局部累加
  for (int i = tid; i < TILE_M*TILE_N; i += blockDim.x) C_s[i] = 0.f;
  __syncthreads();

  // === K 维二分（Stream-K 味道）：两个 CTA 轮换当“生产者”，GMEM->SMEM 并 DSM 广播到对端 ===
  const int num_k_slices = K / TILE_K;   // = 2
  for (int ks = 0; ks < num_k_slices; ++ks) {
    const int k_base = ks * TILE_K;
    const bool i_am_producer = (rank == (ks & 1));

    // 生产者：从 GMEM 装载 A/B 子块到“本 CTA 的 SMEM”
    if (i_am_producer) {
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) {
        int r = i / TILE_K, c = i % TILE_K;
        A_s[i] = A[idx2(row0 + r, k_base + c, ldA)];
      }
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) {
        int r = i / TILE_N, c = i % TILE_N;
        B_s[i] = B[idx2(k_base + r, col0 + c, ldB)];
      }
    }
    __syncthreads();

    // DSM 广播：生产者把“本 CTA 的 SMEM 子块”写到“对端 CTA 的 SMEM”
    cluster.sync();  // 确保装载完成 & 两 CTA 并发存在（DSM 要求）
    if (i_am_producer) {
      // 为了易被模拟器解析，示例用最朴素的逐元素写（也可替换为 cp.async.bulk.shared::cluster+mbarrier）
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) A_peer[i] = A_s[i];
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) B_peer[i] = B_s[i];
      __threadfence_block();  // 保障对端可见（簇同步会再保障时序）
    }
    cluster.sync(); // 确保对端看到最新 DSM 数据

    // 两个 CTA 同时消费 A_s / B_s（各自完成本片的 K 累加）
    for (int lin = tid; lin < TILE_M*TILE_N; lin += blockDim.x) {
      int mi = lin / TILE_N, ni = lin % TILE_N;
      float acc = C_s[lin];
      #pragma unroll
      for (int kk = 0; kk < TILE_K; ++kk) {
        float a = A_s[mi*TILE_K + kk];
        float b = B_s[kk*TILE_N + ni];
        acc = fmaf(a, b, acc);
      }
      C_s[lin] = acc;
    }
    __syncthreads();
  }

  // 片内归并：让 rank==0 从对端 DSM 拉回 C_peer 并与本地 C_s 累加，最后仅 rank==0 写回 GMEM
  cluster.sync();
  if (rank == 0) {
    for (int i = tid; i < TILE_M*TILE_N; i += blockDim.x) C_s[i] += C_peer[i];
  }
  cluster.sync();

  if (rank == 0) {
    for (int lin = tid; lin < TILE_M*TILE_N; lin += blockDim.x) {
      int mi = lin / TILE_N, ni = lin % TILE_N;
      C[idx2(row0 + mi, col0 + ni, ldC)] = C_s[lin];
    }
  }
#endif // __CUDA_ARCH__ >= 900
}

// ======== Host 侧：初始化 + 两种启动方式（<<<>>> 与 cudaLaunchKernelExC）========
void init_mat(float* p, int R, int C, float s) {
  for (int r=0;r<R;++r) for (int c=0;c<C;++c) p[r*C+c] = s * ((r+c)%7 - 3);
}

int main() {
  // 分配与初始化
  float *A,*B,*C;
  cudaMalloc(&A, M*K*sizeof(float));
  cudaMalloc(&B, K*N*sizeof(float));
  cudaMalloc(&C, M*N*sizeof(float));
  {
    float *hA=(float*)malloc(M*K*sizeof(float));
    float *hB=(float*)malloc(K*N*sizeof(float));
    init_mat(hA, M, K, 0.1f);
    init_mat(hB, K, N, 0.2f);
    cudaMemcpy(A, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C, 0, M*N*sizeof(float));
    free(hA); free(hB);
  }

  // 每个 CTA 的动态共享内存字节数
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + TILE_M*TILE_N) * sizeof(float);
  dim3 grid(2,1,1);            // 必须是 clusterSize 的整数倍；这里 2 个 CTA 正好 1 个 cluster
  dim3 block(256,1,1);

#ifndef USE_EX_LAUNCH
  // 方式一：三尖括号（内核上已用 __cluster_dims__(2,1,1) 固定 cluster size）
  streamk_cluster_dsm_gemm_kernel<<<grid, block, smem_bytes>>>(A,B,C, K, N, N);
#else
  // 方式二：CUDA 扩展启动 API（ClusterSim 支持）：cudaLaunchKernelExC
  // 便于在运行时设定 cluster 维度（无需 __cluster_dims__）
  cudaLaunchConfig_t cfg{};
  cfg.gridDim   = grid;
  cfg.blockDim  = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.clusterDim = dim3(2,1,1);     // 设定 cluster 大小（gridDim.x 必须是 2 的倍数）
  void* args[] = { &A, &B, &C, /*ldA*/(void*)&K, /*ldB*/(void*)&N, /*ldC*/(void*)&N };
  cudaLaunchKernelExC(&cfg, (void*)streamk_cluster_dsm_gemm_kernel, args);
#endif

  cudaDeviceSynchronize();

  // 读回和简单校验（打印左上 8×8）
  float hC[64*64];
  cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n");
  for (int r=0;r<8 && r<M;++r) {
    for (int c=0;c<8 && c<N;++c) printf("%7.3f ", hC[r*N + c]);
    printf("\n");
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
