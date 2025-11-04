// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo cluster_sweetspot_no_dsm.cu -o cluster_no_dsm -lcudart
// 运行： ./cluster_no_dsm 2   # 或 4 / 8 / 16

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

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
#define TILE_K 32
#endif
static_assert(M % TILE_M == 0 && N % TILE_N == 0 && K % TILE_K == 0, "dims must be multiples");

__host__ __device__ inline int idx2(int r,int c,int ld){ return r*ld + c; }

// 轻量忙等（避免 nanosleep.u32）
__device__ inline void spin_until_one(volatile int* flag){
  while (atomicAdd((int*)flag, 0) == 0) {
    for (int s=0; s<128; ++s) { asm volatile(""); }
    __threadfence_block();
  }
}

//===================== kernel（模板化 cluster_size；全 GMEM 交互） =====================
template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void gemm_no_dsm_kernel(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int ldA, int ldB, int ldC,
                        // —— 不规则小变量（GMEM）：每个 K-slice 的就绪位 —— //
                        int* __restrict__ slice_ready,   // len = num_k_slices
                        int  ready_len,
                        // —— 大块交换用的 GMEM tile 缓冲（无 DSM）—— //
                        float* __restrict__ A_tile_g,    // len = TILE_M*TILE_K
                        float* __restrict__ B_tile_g,    // len = TILE_K*TILE_N
                        // —— 聚合用 GMEM partials 与其就绪位 —— //
                        float* __restrict__ C_partials_g,// len = CLUSTER_SIZE*rows_per_rank*TILE_N
                        int*   __restrict__ partial_ready,// len = CLUSTER_SIZE
                        int    partial_len)
{
#if __CUDA_ARCH__ >= 900
  const int rank = cg::this_cluster().block_rank();   // 0..CLUSTER_SIZE-1

  // 动态 SMEM：A_s、B_s（每 CTA 私有副本）、C_loc（本 CTA 行段部分和）
  const int rows_per_rank = TILE_M / CLUSTER_SIZE;
  extern __shared__ float smem[];
  float* A_s   = smem;                                // [TILE_M x TILE_K]
  float* B_s   = A_s + TILE_M * TILE_K;               // [TILE_K x TILE_N]
  float* C_loc = B_s + TILE_K * TILE_N;               // [rows_per_rank x TILE_N]

  const int row0 = 0, col0 = 0;
  const int my_row_start = rank * rows_per_rank;
  const int my_row_end   = my_row_start + rows_per_rank;

  // 初始化：清零本 CTA 的 C_loc，并把 partial_ready 清 0（由 rank0 统一完成）
  for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) C_loc[lin] = 0.f;
  if (rank == 0 && threadIdx.x == 0) {
    for (int p = 0; p < CLUSTER_SIZE; ++p) partial_ready[p] = 0;
  }
  __syncthreads(); // CTA 内同步（跨 CTA 依赖 GMEM 标志）

  const int num_k_slices = K / TILE_K;  // 2
  for (int ks = 0; ks < num_k_slices; ++ks) {
    const int producer        = ks % CLUSTER_SIZE;
    const bool i_am_producer  = (rank == producer);
    const int  k_base         = ks * TILE_K;

    // 进入本 slice 前，生产者清 ready=0
    if (i_am_producer && threadIdx.x == 0) { slice_ready[ks] = 0; __threadfence(); }
    __syncthreads();

    // 生产者：GMEM -> 自身 SMEM（A_s/B_s）
    if (i_am_producer) {
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x){
        int r=i/TILE_K, c=i%TILE_K;
        A_s[i] = A[idx2(row0 + r, k_base + c, ldA)];
      }
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){
        int r=i/TILE_N, c=i%TILE_N;
        B_s[i] = B[idx2(k_base + r, col0 + c, ldB)];
      }
      __threadfence_block();
      // —— 无 DSM：把 A_s/B_s 拷到全局 tile 缓冲，供其它 CTA 取用 —— //
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x) A_tile_g[i] = A_s[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x) B_tile_g[i] = B_s[i];
      __threadfence();                    // 先保证 tile 缓冲可见
    }
    __syncthreads();

    // 生产者置 ready=1；其它 CTA 在 GMEM 上自旋等待
    if (i_am_producer && threadIdx.x == 0) { slice_ready[ks] = 1; __threadfence(); }
    if (!i_am_producer && threadIdx.x == 0) { spin_until_one(&slice_ready[ks]); }
    __syncthreads();

    // 非生产者：从 GMEM tile 缓冲把 A/B 拉回自己的 SMEM（无 DSM 的替代）
    if (!i_am_producer) {
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x) A_s[i] = A_tile_g[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x) B_s[i] = B_tile_g[i];
    }
    __syncthreads();

    // 计算：每个 CTA 仅算自己的行段（逻辑不变）
    for (int mi = my_row_start + threadIdx.x; mi < my_row_end; mi += blockDim.x) {
      for (int ni=0; ni<TILE_N; ++ni) {
        float acc = C_loc[(mi - my_row_start) * TILE_N + ni];
        #pragma unroll
        for (int kk=0; kk<TILE_K; ++kk) {
          float a = A_s[mi*TILE_K + kk];
          float b = B_s[kk*TILE_N + ni];
          acc = fmaf(a, b, acc);
        }
        C_loc[(mi - my_row_start) * TILE_N + ni] = acc;
      }
    }
    __syncthreads();
  } // ks

  // —— 聚合阶段：不使用 DSM —— //
  // 非 rank0：将本 CTA 的 C_loc 写入 GMEM partials，并置 partial_ready[rank]=1
  if (rank != 0) {
    float* my_partial = C_partials_g + rank * (rows_per_rank * TILE_N);
    for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x)
      my_partial[lin] = C_loc[lin];
    __threadfence();
    if (threadIdx.x == 0) atomicExch(&partial_ready[rank], 1);
  } else {
    // rank0：先写自己的行段到最终 C，并标记自己 ready（可选）
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
      int r_off = lin / TILE_N, c_off = lin % TILE_N;
      C[idx2(row0 + r_off, col0 + c_off, ldC)] = C_loc[lin];
    }
    __threadfence();
    if (threadIdx.x == 0) atomicExch(&partial_ready[0], 1);

    // 轮询其它 CTA 的 partial_ready，再从 GMEM partials 拉回并写到最终 C
    for (int p=1; p<CLUSTER_SIZE; ++p) {
      if (threadIdx.x == 0) spin_until_one(&partial_ready[p]);
      __syncthreads();
      const float* peer_partial = C_partials_g + p * (rows_per_rank * TILE_N);
      int base_row = p * rows_per_rank;
      for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
        int r_off = lin / TILE_N, c_off = lin % TILE_N;
        C[idx2(row0 + base_row + r_off, col0 + c_off, ldC)] = peer_partial[lin];
      }
      __syncthreads();
    }
  }
#endif
}

//===================== Host：运行时选择 2/4/8/16 =====================
static void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}

template<int CS>
void launch_cs(float* A,float* B,float* C,
               int lda,int ldb,int ldc,
               int* d_ready,int ready_len,
               float* d_A_tile,float* d_B_tile,
               float* d_C_partials,int* d_partial_ready,int partial_len,
               size_t smem_bytes)
{
  dim3 grid(CS,1,1);       // 单 cluster（=CS 个 CTA）
  dim3 block(256,1,1);
  gemm_no_dsm_kernel<CS><<<grid, block, smem_bytes>>>(
      A,B,C, lda,ldb,ldc, d_ready,ready_len, d_A_tile,d_B_tile,
      d_C_partials,d_partial_ready,partial_len);
}

int main(int argc, char** argv)
{
  int cluster_size = 2;              // 可选 2/4/8/16
  if (argc >= 2) cluster_size = atoi(argv[1]);
  if (cluster_size!=2 && cluster_size!=4 && cluster_size!=8 && cluster_size!=16) {
    fprintf(stderr, "[ERROR] cluster_size must be one of {2,4,8,16}\n");
    return 1;
  }
  if (TILE_M % cluster_size != 0) {
    fprintf(stderr, "[ERROR] TILE_M (%d) must be divisible by cluster_size (%d)\n", TILE_M, cluster_size);
    return 1;
  }

  // 分配数据
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

  // GMEM 小变量：per-slice ready
  const int num_k_slices = K / TILE_K;                 // 2
  int* d_ready = nullptr;
  cudaMalloc(&d_ready, num_k_slices * sizeof(int));
  cudaMemset(d_ready, 0, num_k_slices * sizeof(int));

  // GMEM 大块 tile 缓冲（代替 DSM 广播）
  float* d_A_tile = nullptr;
  float* d_B_tile = nullptr;
  cudaMalloc(&d_A_tile, TILE_M*TILE_K * sizeof(float));
  cudaMalloc(&d_B_tile, TILE_K*TILE_N * sizeof(float));

  // GMEM 聚合缓冲与其就绪位（代替 DSM 拉回）
  const int rows_per_rank = TILE_M / cluster_size;
  float* d_C_partials = nullptr;
  int*   d_partial_ready = nullptr;
  cudaMalloc(&d_C_partials, cluster_size * rows_per_rank * TILE_N * sizeof(float));
  cudaMalloc(&d_partial_ready, cluster_size * sizeof(int));
  cudaMemset(d_partial_ready, 0, cluster_size * sizeof(int));

  // 动态 shared：A_s + B_s + C_loc
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + rows_per_rank*TILE_N) * sizeof(float);

  int lda = K, ldb = N, ldc = N;
  switch (cluster_size) {
    case 2:  launch_cs<2 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile,
                           d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 4:  launch_cs<4 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile,
                           d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 8:  launch_cs<8 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile,
                           d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 16: launch_cs<16>(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile,
                           d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
  }
  cudaDeviceSynchronize();

  // 打印左上 8×8
  float hC[64*64];
  cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n");
  for(int r=0;r<8 && r<M;++r){
    for(int c=0;c<8 && c<N;++c) printf("%7.3f ", hC[r*N+c]);
    printf("\n");
  }

  cudaFree(d_partial_ready); cudaFree(d_C_partials);
  cudaFree(d_B_tile); cudaFree(d_A_tile);
  cudaFree(d_ready);
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
