// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo clustersim_streamk_dsm_gemm_sweetspot.cu -o sweetspot_gemm
// 运行示例： ./sweetspot_gemm 4      # cluster_size=4（grid 会按 1 个 cluster 启动）
// 也可 ./sweetspot_gemm              # 默认 cluster_size=2
//
// 说明：
// - 大块数据(A/B) 走 DSM：生产者 CTA 装载到本 SMEM -> DSM 写到各 peer 的 SMEM -> 全簇消费
// - 小变量（cluster 内 work queue / per-slice ready flag）走 GMEM：atomicAdd + 自旋
// - 每个 CTA 只算自己负责的行段，rank 0 用 DSM 拉回 peer 的 C_partial 聚合写回

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

// ====== 可调矩阵/Tile 尺寸（保持整除，便于演示） ======
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
#define TILE_K 32   // K 二分：32 + 32
#endif
static_assert(M % TILE_M == 0 && N % TILE_N == 0 && K % TILE_K == 0, "dims must be multiples");

// ====== 甜蜜区：GMEM 中的 cluster 内小共享变量（cluster-local） ======
#ifndef MAX_CLUSTERS
#define MAX_CLUSTERS 64
#endif
#ifndef MAX_SLICES
#define MAX_SLICES  128
#endif

// 每个 cluster 一个工作队列头（本例调度多个 C-tiles 时使用；本demo默认 1 个tile，也保留此路径供扩展）
__device__ __managed__ int g_cluster_work_head[MAX_CLUSTERS];
// 每个 cluster 的 per-slice “就绪标志”（生产者置 1，消费者自旋等待），演示 GMEM 小共享
__device__ __managed__ int g_slice_ready[MAX_CLUSTERS][MAX_SLICES];

// 线性索引
__host__ __device__ inline int idx2(int r,int c,int ld){ return r*ld + c; }

// ================== Kernel ==================
__global__
void streamk_dsm_gemm_sweetspot_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int ldA, int ldB, int ldC,
                                       int tiles_m, int tiles_n)
{
#if __CUDA_ARCH__ >= 900
  // ---- cluster 句柄 & 运行时 cluster 大小 ----
  cg::cluster_group cluster = cg::this_cluster();
  const int cluster_size = cluster.num_blocks();    // 运行时 clusterDim.x
  const int rank         = cluster.block_rank();    // 0..cluster_size-1

  // ---- 将 grid 划分为若干“连续 block 的 cluster”：cluster_id 假定按 blockIdx.x / cluster_size 划分 ----
  const int cluster_id = blockIdx.x / cluster_size;

  // ---- 一个 cluster 处理多个 C-tiles（本 demo 默认 tiles_m=tiles_n=1，也可加大测试）----
  const int TOTAL_TILES = tiles_m * tiles_n;

  // cluster-local 的工作队列（GMEM）——甜蜜区#1：仅簇内使用的小变量（任务分配）
  // 注意：此处为了展示“GMEM 小共享”，即便 TOTAL_TILES=1 也走同一逻辑
  while (true) {
    int tile_id = atomicAdd(&g_cluster_work_head[cluster_id], 1);  // GMEM 原子：簇内CTA共享
    if (tile_id >= TOTAL_TILES) break;

    // 计算本次 tile 的起始行/列
    int tile_row = (tile_id % tiles_m);
    int tile_col = (tile_id / tiles_m);
    const int row0 = tile_row * TILE_M;
    const int col0 = tile_col * TILE_N;

    // --- 本 tile 的动态共享内存：A_s, B_s 为当前 K-slice，C_s 为“本 CTA 行段”的部分和 ---
    extern __shared__ float smem[];
    float* A_s = smem;                             // [TILE_M x TILE_K]
    float* B_s = A_s + TILE_M * TILE_K;            // [TILE_K x TILE_N]
    float* C_s = B_s + TILE_K * TILE_N;            // [rows_per_rank x TILE_N]（按行分片存放）

    // 行分片：每个 CTA 只负责自己的一段行
    const int rows_per_rank = TILE_M / cluster_size;     // 要求整除
    const int my_row_start  = rank * rows_per_rank;
    const int my_row_end    = my_row_start + rows_per_rank;

    // 对端 SMEM 指针（DSM）——给 rank 0 最终聚合使用；以及广播时的 peer A/B
    // peers 的 C 段地址：按行分片偏移
    float* A_peer = nullptr;
    float* B_peer = nullptr;

    // C 段：把整块 C_s 视作 [TILE_M x TILE_N] 的缓冲，但每个 CTA 只写自己的行段
    // peer 的 C 段首地址 = 其 C_s + peer_row_start*TILE_N
    auto peer_C_ptr = [&](int peer)->float* {
      // 同一个 “smem” 线性区：C_s 从 base = B_s + ...
      float* C_base = C_s;
      return cluster.map_shared_rank(C_base, peer) + (peer * rows_per_rank) * TILE_N;
    };

    // 初始化本 CTA 的行段为 0
    for (int lin = threadIdx.x; lin < rows_per_rank * TILE_N; lin += blockDim.x) {
      C_s[lin] = 0.f;
    }
    __syncthreads();

    // ---- K-slices（Stream-K 风味）：每片由“生产者 CTA = ks % cluster_size”装载并 DSM 广播 ----
    const int num_k_slices = K / TILE_K; // e.g., 2
    for (int ks = 0; ks < num_k_slices; ++ks) {

      const int producer = ks % cluster_size;
      const bool i_am_producer = (rank == producer);
      const int k_base = ks * TILE_K;

      // 生产者：GMEM -> 自己 SMEM
      if (i_am_producer) {
        for (int i = threadIdx.x; i < TILE_M*TILE_K; i += blockDim.x) {
          int r = i / TILE_K, c = i % TILE_K;
          A_s[i] = A[idx2(row0 + r, k_base + c, ldA)];
        }
        for (int i = threadIdx.x; i < TILE_K*TILE_N; i += blockDim.x) {
          int r = i / TILE_N, c = i % TILE_N;
          B_s[i] = B[idx2(k_base + r, col0 + c, ldB)];
        }
        __threadfence_block();
      }
      __syncthreads();

      // 甜蜜区#2：GMEM 的“片就绪标志”——仅簇内使用，但放 GMEM（不规则小变量）
      if (i_am_producer && threadIdx.x == 0) {
        g_slice_ready[cluster_id][ks] = 1;    // 生产者置位
        __threadfence();                      // GMEM 可见性
      }
      if (!i_am_producer && threadIdx.x == 0) {
        // 消费者自旋等待 GMEM flag（即使我们也会 cluster.sync()，这里故意制造 GMEM 小通信路径）
        while (atomicAdd(&g_slice_ready[cluster_id][ks], 0) == 0) { /* spin */ }
      }
      __syncthreads();

      // DSM 广播：生产者把本 CTA 的 A_s/B_s 写到每个 peer 的 SMEM
      // （演示大块数据走 DSM；在真实代码里可用 cp.async.bulk.shared::cluster + mbarrier）
      cluster.sync();  // DSM 访问保护
      if (i_am_producer) {
        for (int p = 0; p < cluster_size; ++p) {
          if (p == producer) continue;
          A_peer = cluster.map_shared_rank(A_s, p);
          B_peer = cluster.map_shared_rank(B_s, p);
          for (int i = threadIdx.x; i < TILE_M*TILE_K; i += blockDim.x) A_peer[i] = A_s[i];
          for (int i = threadIdx.x; i < TILE_K*TILE_N; i += blockDim.x) B_peer[i] = B_s[i];
        }
      }
      cluster.sync();  // 确保各 peer 看到最新 DSM 数据

      // 全簇消费：每个 CTA 仅计算“自己的行段”（避免重复计算）
      for (int mi = my_row_start + threadIdx.x; mi < my_row_end; mi += blockDim.x) {
        for (int ni = 0; ni < TILE_N; ++ni) {
          float acc = C_s[(mi - my_row_start) * TILE_N + ni];
          #pragma unroll
          for (int kk = 0; kk < TILE_K; ++kk) {
            float a = A_s[mi * TILE_K + kk];      // DSM 广播/本地装载的 A 子块
            float b = B_s[kk * TILE_N + ni];      // DSM 广播/本地装载的 B 子块
            acc = fmaf(a, b, acc);
          }
          C_s[(mi - my_row_start) * TILE_N + ni] = acc;
        }
      }
      __syncthreads();
    } // end ks

    // ---- 最终聚合：rank 0 用 DSM 从各 peer 拉回其行段，拼成整块 C 并写回 GMEM ----
    cluster.sync();
    if (rank == 0) {
      // rank 0 已计算自己的 [0:rows_per_rank)；从每个 peer 拉回其 [peer*rows_per_rank:(peer+1)*rows_per_rank)
      for (int p = 1; p < cluster_size; ++p) {
        float* C_peer_rows = peer_C_ptr(p);  // DSM 指向 peer 的 C 段首地址
        // 逐元素拷到 rank 0 的 C_s 后半区域（按行段排）
        for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) {
          int dst_off = p * rows_per_rank * TILE_N + lin;
          C_s[dst_off] = C_peer_rows[lin];
        }
        __syncthreads();
      }
      // rank 0 写回整个 TILE_M×TILE_N 的 C tile
      for (int lin = threadIdx.x; lin < TILE_M*TILE_N; lin += blockDim.x) {
        int mi = lin / TILE_N, ni = lin % TILE_N;
        C[idx2(row0 + mi, col0 + ni, ldC)] = C_s[lin];
      }
    }
    cluster.sync();

    // 复位 GMEM 小标志（便于下一个 tile 循环复用）
    if (rank == 0 && threadIdx.x == 0) {
      for (int ks = 0; ks < num_k_slices; ++ks) g_slice_ready[cluster_id][ks] = 0;
    }
    cluster.sync();
  } // while tiles
#endif // __CUDA_ARCH__ >= 900
}

// =============== Host 侧：初始化 + 运行时 cluster 设定（cudaLaunchKernelExC） ===============
void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}

int main(int argc, char** argv)
{
  int cluster_size = 2;                     // 运行时可调：clusterDim.x
  if (argc >= 2) cluster_size = atoi(argv[1]);
  if (cluster_size < 1) cluster_size = 1;

  // 一次只跑 1 个 cluster（便于观察簇内行为）
  const int num_clusters = 1;
  dim3 grid(cluster_size * num_clusters, 1, 1);
  dim3 block(256, 1, 1);

  // 初始化矩阵
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

  // 清零 GMEM 小共享（甜蜜区）
  cudaMemset(g_cluster_work_head, 0, sizeof(g_cluster_work_head));
  cudaMemset(g_slice_ready, 0, sizeof(g_slice_ready));

  // 动态 shared：A_s + B_s + C_s（注意 C_s 这里为整 tile 大小，便于 rank0 聚合）
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + TILE_M*TILE_N) * sizeof(float);

  // 运行时 cluster 启动（ClusterSim 支持）：cudaLaunchKernelExC
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.clusterDim = dim3(cluster_size, 1, 1);   // 运行时设定 cluster 大小（关键）

  int tiles_m = M / TILE_M;   // 默认 1
  int tiles_n = N / TILE_N;   // 默认 1

  void* args[] = { &A, &B, &C, (void*)&K, (void*)&N, (void*)&N, &tiles_m, &tiles_n };
  cudaLaunchKernelExC(&cfg, (void*)streamk_dsm_gemm_sweetspot_kernel, args);
  cudaDeviceSynchronize();

  // 打印左上 8×8
  float hC[64*64];
  cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n");
  for(int r=0;r<8 && r<M;++r){
    for(int c=0;c<8 && c<N;++c) printf("%7.3f ", hC[r*N+c]);
    printf("\n");
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
