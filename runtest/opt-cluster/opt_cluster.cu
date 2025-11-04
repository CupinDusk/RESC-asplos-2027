// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo cluster_sweetspot_dsm_pull.cu -o cluster_dsm_pull -lcudart
// 运行： ./cluster_dsm_pull 2   # 或 4 / 8 / 16

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

// ---- 尺寸（微基准，小算子，整除约束）----
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
#define TILE_K 32      // 两个 k-slices
#endif
static_assert(M%TILE_M==0 && N%TILE_N==0 && K%TILE_K==0, "dims must be multiples");

__host__ __device__ inline int idx2(int r,int c,int ld){ return r*ld + c; }

// 无 nanosleep 的轻量忙等（避免 PTX nanosleep.u32）
__device__ inline void spin_until_one(volatile int* flag){
  while (atomicAdd((int*)flag, 0) == 0) {
    // 给编译器/调度器一点活性，不生成 nanosleep
    #pragma unroll 4
    for (int s=0;s<64;++s) asm volatile("");
    __threadfence_block();
  }
}

//================ kernel（模板化 cluster_size；DSM = PULL 远程读） ================
template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void gemm_dsm_pull_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int ldA, int ldB, int ldC,
                          // —— 不规则小变量：GMEM，就绪位 —— //
                          int* __restrict__ ready,  // len = num_k_slices
                          int  ready_len)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();                 // 0..CLUSTER_SIZE-1

  // 动态 SMEM：A_s / B_s 是“生产者的本地缓冲”；C_loc 是本 CTA 行段的累加
  const int rows_per_rank = TILE_M / CLUSTER_SIZE;
  extern __shared__ float smem[];
  float* A_s   = smem;                                   // [TILE_M x TILE_K]
  float* B_s   = A_s + TILE_M * TILE_K;                  // [TILE_K x TILE_N]
  float* C_loc = B_s + TILE_K * TILE_N;                  // [rows_per_rank x TILE_N]

  // 单 tile 微基准：(row0,col0)=(0,0)
  const int row0 = 0, col0 = 0;
  const int my_row_start = rank * rows_per_rank;
  const int my_row_end   = my_row_start + rows_per_rank;

  // 清零本 CTA 的行段
  for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) C_loc[lin] = 0.f;
  __syncthreads();

  const int num_k_slices = K / TILE_K;   // 2
  for (int ks=0; ks<num_k_slices; ++ks) {
    const int producer       = ks % CLUSTER_SIZE;
    const bool i_am_producer = (rank == producer);
    const int  k_base        = ks * TILE_K;

    // —— 生产者加载 A/B 到“自己 SMEM”（不再复制给 peers）——
    if (i_am_producer) {
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x){
        int r=i/TILE_K, c=i%TILE_K;
        A_s[i] = A[idx2(row0 + r, k_base + c, ldA)];
      }
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){
        int r=i/TILE_N, c=i%TILE_N;
        B_s[i] = B[idx2(k_base + r, col0 + c, ldB)];
      }
      __threadfence_block();                    // 生产者本地可见
    }
    __syncthreads();

    // —— GMEM 小就绪位：生产者置 1；消费者等待 ——（保留为“甜蜜区小变量”）
    if (i_am_producer && threadIdx.x==0) { ready[ks] = 1; __threadfence(); }
    if (!i_am_producer && threadIdx.x==0) spin_until_one(&ready[ks]);
    __syncthreads();

    // —— DSM 可见性屏障：确保远程 SMEM 可被其它 CTA 看见 ——（CUDA 指南推荐）
    cluster.sync();

    // —— 每个 CTA 计算自己的行段：A/B 源指针 = 本地（若是生产者）或远程映射（若是消费者）——
    // A_src/B_src 指向“生产者 CTA 的 A_s/B_s”
    float* A_src = i_am_producer ? A_s : cluster.map_shared_rank(A_s, producer);
    float* B_src = i_am_producer ? B_s : cluster.map_shared_rank(B_s, producer);

    for (int mi = my_row_start + threadIdx.x; mi < my_row_end; mi += blockDim.x) {
      for (int ni=0; ni<TILE_N; ++ni) {
        float acc = C_loc[(mi - my_row_start) * TILE_N + ni];
        #pragma unroll
        for (int kk=0; kk<TILE_K; ++kk) {
          float a = A_src[mi*TILE_K + kk];       // 生产者本地读 / 消费者远程读
          float b = B_src[kk*TILE_N + ni];
          acc = fmaf(a, b, acc);
        }
        C_loc[(mi - my_row_start) * TILE_N + ni] = acc;
      }
    }
    __syncthreads();
  } // ks

  // —— 聚合：rank 0 远程读取 peers 的 C_loc（DSM PULL），直接写回 GMEM —— 
  cluster.sync();
  if (rank == 0) {
    // rank0 自己的行段
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
      int r_off = lin / TILE_N, c_off = lin % TILE_N;
      C[idx2(row0 + r_off, col0 + c_off, ldC)] = C_loc[lin];
    }
    __syncthreads();
    // 其它 peers：直接“远程读”它们的 C_loc
    for (int p=1; p<CLUSTER_SIZE; ++p) {
      float* C_peer_loc = cluster.map_shared_rank(C_loc, p);
      int base_row = p * rows_per_rank;
      for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
        int r_off = lin / TILE_N, c_off = lin % TILE_N;
        C[idx2(row0 + base_row + r_off, col0 + c_off, ldC)] = C_peer_loc[lin];
      }
      __syncthreads();
    }
  }
  cluster.sync();
#endif
}

//================ Host：运行时选择 2/4/8/16（与原版一致） ==================
static void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}

template<int CS>
void launch_cs(float* A,float* B,float* C,
               int lda,int ldb,int ldc,
               int* d_ready,int ready_len,
               size_t smem_bytes)
{
  dim3 grid(CS,1,1);          // 单 cluster（=CS 个 CTA）
  dim3 block(256,1,1);
  gemm_dsm_pull_kernel<CS><<<grid, block, smem_bytes>>>(A,B,C, lda,ldb,ldc, d_ready, ready_len);
}

int main(int argc, char** argv)
{
  int cluster_size = 2;
  if (argc>=2) cluster_size = atoi(argv[1]);
  if (cluster_size!=2 && cluster_size!=4 && cluster_size!=8 && cluster_size!=16) {
    fprintf(stderr,"[ERROR] cluster_size must be one of {2,4,8,16}\n");
    return 1;
  }
  if (TILE_M % cluster_size != 0) {
    fprintf(stderr,"[ERROR] TILE_M (%d) must be divisible by cluster_size (%d)\n", TILE_M, cluster_size);
    return 1;
  }

  // 数据
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

  // GMEM 小变量：per-slice 就绪位
  const int num_k_slices = K / TILE_K;   // 2
  int* d_ready = nullptr;
  cudaMalloc(&d_ready, num_k_slices * sizeof(int));
  cudaMemset(d_ready, 0, num_k_slices * sizeof(int));

  // 动态 shared：A_s + B_s + C_loc
  const int rows_per_rank = TILE_M / cluster_size;
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + rows_per_rank*TILE_N) * sizeof(float);

  int lda=K, ldb=N, ldc=N;
  switch (cluster_size) {
    case 2:  launch_cs<2 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 4:  launch_cs<4 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 8:  launch_cs<8 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 16: launch_cs<16>(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
  }
  cudaDeviceSynchronize();

  // 验证输出（左上 8×8）
  float hC[64*64];
  cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n");
  for(int r=0;r<8 && r<M;++r){
    for(int c=0;c<8 && c<N;++c) printf("%7.3f ", hC[r*N+c]);
    printf("\n");
  }

  cudaFree(d_ready); cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
