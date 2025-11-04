// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo cluster_sweetspot_dsm_gmem_fix.cu -o cluster_sweetspot -lcudart
// 运行： ./cluster_sweetspot 2   # 或 4 / 8 / 16

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

// --------- 小工具：无 nanosleep 的温和忙等（避免生成 nanosleep.u32）---------
__device__ inline void spin_until_one(volatile int* flag){
  // 读 flag==1 才继续；每轮做一点点“无副作用活性”避免被编译器全去掉
  while (atomicAdd((int*)flag, 0) == 0) {
    // 轻量忙等：做几次空转 + 局部 fence，避免 nanosleep 指令
    for (int s=0; s<128; ++s) { asm volatile(""); }
    __threadfence_block();
  }
}

//===================== kernel（模板化 cluster_size） =====================
template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void gemm_sweetspot_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int ldA, int ldB, int ldC,
                           // —— 不规则小变量：全局内存 GMEM —— //
                           int* __restrict__ ready,   // len = num_k_slices（本 demo 单 cluster）
                           int  ready_len)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();                 // 0..CLUSTER_SIZE-1

  // 动态 SMEM：
  //  A_s: TILE_M x TILE_K（全 tile）
  //  B_s: TILE_K x TILE_N（全 tile）
  //  C_loc: rows_per_rank x TILE_N（本 CTA 局部行段的部分和）
  const int rows_per_rank = TILE_M / CLUSTER_SIZE;
  extern __shared__ float smem[];
  float* A_s   = smem;
  float* B_s   = A_s + TILE_M * TILE_K;
  float* C_loc = B_s + TILE_K * TILE_N;

  const int row0 = 0, col0 = 0;
  const int my_row_start = rank * rows_per_rank;
  const int my_row_end   = my_row_start + rows_per_rank;

  // 清零本 CTA 的行段
  for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) C_loc[lin] = 0.f;
  __syncthreads();

  const int num_k_slices = K / TILE_K;  // 2
  for (int ks = 0; ks < num_k_slices; ++ks) {
    const int producer        = ks % CLUSTER_SIZE;
    const bool i_am_producer  = (rank == producer);
    const int  k_base         = ks * TILE_K;

    // ---- GMEM 小变量：进入前复位 & 就绪同步 ----
    if (i_am_producer && threadIdx.x == 0) { ready[ks] = 0; __threadfence(); }
    __syncthreads();

    // 生产者：GMEM -> 自身 SMEM（A_s/B_s 大块）
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
    }
    __syncthreads();

    // 生产者置 ready；消费者在 GMEM 自旋（无 nanosleep）
    if (i_am_producer && threadIdx.x == 0) { ready[ks] = 1; __threadfence(); }
    if (!i_am_producer && threadIdx.x == 0) { spin_until_one(&ready[ks]); }
    __syncthreads();

    // ---- 大块数据走 DSM：把生产者的 A_s/B_s 写到所有 peer 的 SMEM ----
    cluster.sync();
    if (i_am_producer) {
      for (int p=0; p<CLUSTER_SIZE; ++p){
        if (p==producer) continue;
        float* A_peer = cluster.map_shared_rank(A_s, p);
        float* B_peer = cluster.map_shared_rank(B_s, p);
        for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x) A_peer[i]=A_s[i];
        for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x) B_peer[i]=B_s[i];
      }
    }
    cluster.sync(); // DSM 广播完成

    // ---- 计算：每个 CTA 仅算自己的行段 ----
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

  // ---- 簇内聚合：rank 0 DSM 拉回 peers 的 C_loc 并写回 GMEM ----
  cluster.sync();
  if (rank == 0) {
    // rank0 自己的行段
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
      int r_off = lin / TILE_N, c_off = lin % TILE_N;
      C[idx2(row0 + r_off, col0 + c_off, ldC)] = C_loc[lin];
    }
    __syncthreads();
    // 其余 peers
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

//===================== Host：运行时选择 2/4/8/16 =====================
static void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}

template<int CS>
void launch_cs(float* A,float* B,float* C,
               int lda,int ldb,int ldc,
               int* d_ready,int ready_len,
               size_t smem_bytes)
{
  dim3 grid(CS,1,1);       // 单 cluster（=CS 个 CTA）
  dim3 block(256,1,1);
  gemm_sweetspot_kernel<CS><<<grid, block, smem_bytes>>>(A,B,C, lda,ldb,ldc, d_ready, ready_len);
}

int main(int argc, char** argv)
{
  int cluster_size = 2;               // 可选 2/4/8/16
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
  const int num_k_slices = K / TILE_K;  // 2
  int* d_ready = nullptr;
  cudaMalloc(&d_ready, num_k_slices * sizeof(int));
  cudaMemset(d_ready, 0, num_k_slices * sizeof(int));

  // 动态 shared：|A_s| + |B_s| + |C_loc|
  const int rows_per_rank = TILE_M / cluster_size;
  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + rows_per_rank*TILE_N) * sizeof(float);

  int lda = K, ldb = N, ldc = N;
  switch (cluster_size) {
    case 2:  launch_cs<2 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 4:  launch_cs<4 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 8:  launch_cs<8 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 16: launch_cs<16>(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
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

  cudaFree(d_ready); cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
