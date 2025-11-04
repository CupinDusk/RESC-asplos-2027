// nvcc -std=c++17 -arch=sm_90a streamk_cluster_dsm_gemm.cu -o streamk_cluster_dsm_gemm
// 运行: ./streamk_cluster_dsm_gemm
// 可改编: M,N,K,TILE_M,TILE_N,TILE_K, 以及 grid 配置

#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ========== 可调参数 ==========
#ifndef M
#define M 32
#endif
#ifndef N
#define N 32
#endif
#ifndef K
#define K 32
#endif

// tile 尺寸（要能被 M,N,K 整除以简化样例）
#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 16    // K 被分成 2 片: 16+16
#endif

// 如果你的模拟器暂不支持 DSMEM，可把此宏置 1，用 GMEM 模拟跨 CTA 通信（功能等价，性能不代表）
#ifndef USE_FAKE_DSM
#define USE_FAKE_DSM 0
#endif

// 简单的 index 宏
__host__ __device__ inline
int idx2(int r, int c, int ld) { return r*ld + c; }

// 2-CTA cluster
__global__ __cluster_dims__(2,1,1)
void streamk_cluster_dsm_gemm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
#if USE_FAKE_DSM
                                     // 假 DSM（GMEM）缓冲（仅在 USE_FAKE_DSM=1 用到）
                                     float* __restrict__ gA_tile,
                                     float* __restrict__ gB_tile,
                                     float* __restrict__ gC_partial,
#endif
                                     int ldA, int ldB, int ldC) {
#if (__CUDA_ARCH__ >= 900) || USE_FAKE_DSM
  // 一个 cluster 有 2 个 CTA：rank 0 / 1
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();         // 0 or 1
  int peer = rank ^ 1;

  // 让这个 cluster 只负责 C 的 (0:TILE_M, 0:TILE_N) 这个 tile（微基准最小化）
  const int row0 = 0;
  const int col0 = 0;

  // 共享内存布局：
  //  - A_s: TILE_M x TILE_K
  //  - B_s: TILE_K x TILE_N
  //  - C_s: TILE_M x TILE_N  (存放本 CTA 的部分和)
  extern __shared__ float smem[];
  float* A_s = smem;                                           // TILE_M * TILE_K
  float* B_s = A_s + TILE_M * TILE_K;                          // TILE_K * TILE_N
  float* C_s = B_s + TILE_K * TILE_N;                          // TILE_M * TILE_N

  // 对端 CTA 的 SMEM 指针（DSM 地址）
  float* A_peer = nullptr;
  float* B_peer = nullptr;
  float* C_peer = nullptr;

#if !USE_FAKE_DSM
  A_peer = cluster.map_shared_rank(A_s, peer);
  B_peer = cluster.map_shared_rank(B_s, peer);
  C_peer = cluster.map_shared_rank(C_s, peer);
#endif

  // C 的局部寄存器累加：每线程做若干 (m,n) 元素
  // 这里简单起见让每线程算 4 个输出元素（行向量展开）
  const int tid = threadIdx.x;
  const int elems_per_thread = (TILE_M*TILE_N + blockDim.x - 1)/blockDim.x;
  // 每线程负责 elems_per_thread 个 (mi, ni) 线性位置
  // 为避免越界，做边界判断
  // 初始化本 CTA 的 C_s = 0
  for (int e = 0; e < elems_per_thread; ++e) {
    int linear = tid + e*blockDim.x;
    if (linear < TILE_M*TILE_N) C_s[linear] = 0.f;
  }
  __syncthreads();

  // K 被拆成两片： [0, TILE_K) 由 rank==0 装载并广播； [TILE_K, 2*TILE_K) 由 rank==1 装载并广播
  const int num_k_slices = K / TILE_K;     // 期望=2
  for (int ks = 0; ks < num_k_slices; ++ks) {
    int k_base = ks * TILE_K;

    // 谁是本轮的“生产者 CTA”
    bool i_am_producer = (rank == (ks & 1));

    // === 生产者把 GMEM -> 自己 SMEM ===
    if (i_am_producer) {
      // 装 A 子块 (TILE_M x TILE_K)
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) {
        int r = i / TILE_K;
        int c = i % TILE_K;
        A_s[i] = A[idx2(row0 + r, k_base + c, ldA)];
      }
      // 装 B 子块 (TILE_K x TILE_N)
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) {
        int r = i / TILE_N;
        int c = i % TILE_N;
        B_s[i] = B[idx2(k_base + r, col0 + c, ldB)];
      }
    }
    __syncthreads();

    // === 生产者把本 CTA 的 SMEM 子块“广播”给对端 CTA 的 SMEM（DSM 写）===
#if !USE_FAKE_DSM
    cluster.sync(); // 保证生产者已完成装载
    if (i_am_producer) {
      // 简单逐元素拷贝（演示 DSM 写；生产环境可用 cp.async.bulk）
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) A_peer[i] = A_s[i];
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) B_peer[i] = B_s[i];
    }
    cluster.sync(); // 保证对端看到最新的 SMEM
#else
    // 假 DSM：用 GMEM 缓冲模拟跨 CTA 广播（功能一致，性能不代表）
    if (i_am_producer) {
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) gA_tile[i] = A_s[i];
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) gB_tile[i] = B_s[i];
    }
    __syncthreads();
    if (!i_am_producer) {
      for (int i = tid; i < TILE_M*TILE_K; i += blockDim.x) A_s[i] = gA_tile[i];
      for (int i = tid; i < TILE_K*TILE_N; i += blockDim.x) B_s[i] = gB_tile[i];
    }
    __syncthreads();
#endif

    // === 两个 CTA 同时消费本轮 A_s / B_s，累加到各自 C_s（各自负责 K 的半边，总和等价 Stream-K）===
    for (int e = 0; e < elems_per_thread; ++e) {
      int linear = tid + e*blockDim.x;
      if (linear >= TILE_M*TILE_N) break;
      int mi = linear / TILE_N;
      int ni = linear % TILE_N;

      float acc = C_s[linear];
      // 朴素 GEMM：按本片 K 范围累加
      for (int kk = 0; kk < TILE_K; ++kk) {
        float a = A_s[mi*TILE_K + kk];
        float b = B_s[kk*TILE_N + ni];
        acc = fmaf(a, b, acc);
      }
      C_s[linear] = acc;
    }
    __syncthreads();
  }

  // === 片内归并：让 rank==0 用 DSM 从 rank==1 拉取 C_s 并累加，最后写回 GMEM ===
#if !USE_FAKE_DSM
  C_peer = cluster.map_shared_rank(C_s, peer);
  cluster.sync();

  if (rank == 0) {
    // 把对端的 C_partial 加到本地
    for (int i = tid; i < TILE_M*TILE_N; i += blockDim.x) {
      C_s[i] += C_peer[i];
    }
  }
  cluster.sync();
#else
  // 假 DSM：把 peer 的 C_s 先抄到 GMEM，再由 0 号 CTA 拉回累加
  if (rank == 1) {
    for (int i = tid; i < TILE_M*TILE_N; i += blockDim.x) gC_partial[i] = C_s[i];
  }
  __syncthreads();
  if (rank == 0) {
    for (int i = tid; i < TILE_M*TILE_N; i += blockDim.x) C_s[i] += gC_partial[i];
  }
  __syncthreads();
#endif

  // === 最终写回 GMEM（只让 rank==0 写，避免重复）===
  if (rank == 0) {
    for (int e = 0; e < elems_per_thread; ++e) {
      int linear = tid + e*blockDim.x;
      if (linear >= TILE_M*TILE_N) break;
      int mi = linear / TILE_N;
      int ni = linear % TILE_N;
      C[idx2(row0 + mi, col0 + ni, ldC)] = C_s[linear];
    }
  }
#endif // arch guard
}

void init_mat(float* p, int rows, int cols, float vscale) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      p[r*cols + c] = vscale * ((r + c) % 7 - 3); // 小整数模式，利于核对
}

int main() {
  const int bytesA = M*K*sizeof(float);
  const int bytesB = K*N*sizeof(float);
  const int bytesC = M*N*sizeof(float);

  float *hA = (float*)malloc(bytesA);
  float *hB = (float*)malloc(bytesB);
  float *hC = (float*)malloc(bytesC);
  init_mat(hA, M, K, 0.1f);
  init_mat(hB, K, N, 0.2f);
  memset(hC, 0, bytesC);

  float *A, *B, *C;
  cudaMalloc(&A, bytesA);
  cudaMalloc(&B, bytesB);
  cudaMalloc(&C, bytesC);
  cudaMemcpy(A, hA, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB, bytesB, cudaMemcpyHostToDevice);
  cudaMemset(C, 0, bytesC);

#if USE_FAKE_DSM
  // 假 DSM 缓冲区
  float *gA_tile, *gB_tile, *gC_partial;
  cudaMalloc(&gA_tile, TILE_M*TILE_K*sizeof(float));
  cudaMalloc(&gB_tile, TILE_K*TILE_N*sizeof(float));
  cudaMalloc(&gC_partial, TILE_M*TILE_N*sizeof(float));
#endif

  // 一个 cluster (=2 CTAs) 处理 1 个 C tile（微基准）
  dim3 grid(1,1,1);
  dim3 block(256,1,1);
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + TILE_M*TILE_N)*sizeof(float);

#if USE_FAKE_DSM
  streamk_cluster_dsm_gemm_kernel<<<grid, block, smem_bytes>>>(A, B, C, gA_tile, gB_tile, gC_partial, K, N, N);
#else
  streamk_cluster_dsm_gemm_kernel<<<grid, block, smem_bytes>>>(A, B, C, K, N, N);
#endif
  cudaDeviceSynchronize();

  cudaMemcpy(hC, C, bytesC, cudaMemcpyDeviceToHost);

  // 简单打印 8x8 左上角
  printf("C[0:8,0:8]:\n");
  for (int r = 0; r < min(8, M); ++r) {
    for (int c = 0; c < min(8, N); ++c) {
      printf("%7.3f ", hC[r*N + c]);
    }
    printf("\n");
  }

#if USE_FAKE_DSM
  cudaFree(gA_tile); cudaFree(gB_tile); cudaFree(gC_partial);
#endif
  cudaFree(A); cudaFree(B); cudaFree(C);
  free(hA); free(hB); free(hC);
  return 0;
}
