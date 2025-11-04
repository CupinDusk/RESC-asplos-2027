// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo nocluster.cu -o nocluster -lcudart
// 运行: ./nocluster 8   # cluster_size 可选 2/4/8/16（同含义：协作 CTA 数，不过走 L2/GMEM）

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef M
#define M 128
#endif
#ifndef N
#define N 128
#endif
#ifndef K
#define K 128
#endif
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
static_assert(M%TILE_M==0 && N%TILE_N==0 && K%TILE_K==0, "dims must be multiples");

__host__ __device__ inline int idx2(int r,int c,int ld){ return r*ld + c; }
__device__ inline void spin_until_one(volatile int* flag){
  while (atomicAdd((int*)flag, 0) == 0) {
    #pragma unroll 4
    for (int s=0; s<64; ++s) asm volatile("");
    __threadfence_block();
  }
}

template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void gemm_nodsm_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int ldA, int ldB, int ldC,
                       int* __restrict__ slice_ready, int ready_len,
                       float* __restrict__ A_tile_g, float* __restrict__ B_tile_g,
                       float* __restrict__ C_partials_g, int* __restrict__ partial_ready, int partial_len)
{
#if __CUDA_ARCH__ >= 900
  const int rank = cg::this_cluster().block_rank();
  const int rows_per_rank = TILE_M / CLUSTER_SIZE;

  extern __shared__ float smem[];
  float* A_s   = smem;                                  // [TILE_M x TILE_K]
  float* B_s   = A_s + TILE_M * TILE_K;                 // [TILE_K x TILE_N]
  float* C_loc = B_s + TILE_K * TILE_N;                 // [rows_per_rank x TILE_N]

  const int row0=0, col0=0;
  const int my_row_start = rank * rows_per_rank;
  const int my_row_end   = my_row_start + rows_per_rank;

  for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) C_loc[lin]=0.f;
  if (rank==0 && threadIdx.x==0) for(int p=0;p<CLUSTER_SIZE;++p) partial_ready[p]=0;
  __syncthreads();

  const int num_k_slices = K / TILE_K;  // 2
  for (int ks=0; ks<num_k_slices; ++ks) {
    const int producer       = ks % CLUSTER_SIZE;
    const bool i_am_producer = (rank == producer);
    const int  k_base        = ks * TILE_K;

    if (i_am_producer && threadIdx.x==0) { slice_ready[ks]=0; __threadfence(); }
    __syncthreads();

    // 生产者：GMEM -> 本地 SMEM
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
      // 无 DSM：把本地 SMEM 的 A_s/B_s 拷到 GMEM 的 tile 缓冲
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x) A_tile_g[i]=A_s[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x) B_tile_g[i]=B_s[i];
      __threadfence();
    }
    __syncthreads();

    // 生产者置就绪；消费者在 GMEM 等待
    if (i_am_producer && threadIdx.x==0) { slice_ready[ks]=1; __threadfence(); }
    if (!i_am_producer && threadIdx.x==0) spin_until_one(&slice_ready[ks]);
    __syncthreads();

    // 消费者：从 GMEM tile 缓冲拷回到自己的 SMEM
    if (!i_am_producer) {
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x) A_s[i]=A_tile_g[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x) B_s[i]=B_tile_g[i];
    }
    __syncthreads();

    // 计算（同 DSM 版）
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
  }

  // 聚合：无 DSM 版把各 CTA 的 C_loc 写 GMEM partials，rank0 拉回写最终 C
  if (rank != 0) {
    float* my_partial = C_partials_g + rank * (rows_per_rank * TILE_N);
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) my_partial[lin]=C_loc[lin];
    __threadfence();
    if (threadIdx.x==0) atomicExch(&partial_ready[rank],1);
  } else {
    // rank0 自身
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
      int r_off=lin/TILE_N, c_off=lin%TILE_N;
      C[idx2(row0 + r_off, col0 + c_off, ldC)] = C_loc[lin];
    }
    __threadfence();
    if (threadIdx.x==0) atomicExch(&partial_ready[0],1);
    // 拉回 peers
    for (int p=1; p<CLUSTER_SIZE; ++p) {
      if (threadIdx.x==0) spin_until_one(&partial_ready[p]);
      __syncthreads();
      const float* peer_partial = C_partials_g + p * (rows_per_rank * TILE_N);
      int base_row = p * rows_per_rank;
      for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
        int r_off=lin/TILE_N, c_off=lin%TILE_N;
        C[idx2(row0 + base_row + r_off, col0 + c_off, ldC)] = peer_partial[lin];
      }
      __syncthreads();
    }
  }
#endif
}

// ---- Host ----
static void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}
template<int CS>
void launch_cs(float* A,float* B,float* C,int lda,int ldb,int ldc,
               int* d_ready,int ready_len,
               float* d_A_tile,float* d_B_tile,
               float* d_C_partials,int* d_partial_ready,int partial_len,
               size_t smem_bytes)
{
  dim3 grid(CS,1,1), block(256,1,1);
  gemm_nodsm_kernel<CS><<<grid, block, smem_bytes>>>(A,B,C, lda,ldb,ldc,
      d_ready,ready_len, d_A_tile,d_B_tile, d_C_partials,d_partial_ready,partial_len);
}

int main(int argc, char** argv){
  int cluster_size = 8;
  if (argc>=2) cluster_size = atoi(argv[1]);
  if (!(cluster_size==2||cluster_size==4||cluster_size==8||cluster_size==16)) { printf("cluster_size must be 2/4/8/16\n"); return 1; }
  if (TILE_M % cluster_size) { printf("TILE_M (%d) must be divisible by cluster_size (%d)\n",TILE_M,cluster_size); return 1; }

  float *A,*B,*C;
  cudaMalloc(&A, M*K*sizeof(float));
  cudaMalloc(&B, K*N*sizeof(float));
  cudaMalloc(&C, M*N*sizeof(float));
  {
    float *hA=(float*)malloc(M*K*sizeof(float));
    float *hB=(float*)malloc(K*N*sizeof(float));
    init_mat(hA, M, K, 0.1f); init_mat(hB, K, N, 0.2f);
    cudaMemcpy(A,hA,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B,hB,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(C,0,M*N*sizeof(float));
    free(hA); free(hB);
  }

  // 小变量
  const int num_k_slices = K / TILE_K;  // 2
  int* d_ready=nullptr; cudaMalloc(&d_ready, num_k_slices*sizeof(int)); cudaMemset(d_ready,0,num_k_slices*sizeof(int));

  // GMEM tile 缓冲（大块）
  float *d_A_tile=nullptr, *d_B_tile=nullptr;
  cudaMalloc(&d_A_tile, TILE_M*TILE_K*sizeof(float));
  cudaMalloc(&d_B_tile, TILE_K*TILE_N*sizeof(float));

  // GMEM partials 与其就绪位
  const int rows_per_rank = TILE_M / cluster_size;
  float* d_C_partials=nullptr; int* d_partial_ready=nullptr;
  cudaMalloc(&d_C_partials, cluster_size * rows_per_rank * TILE_N * sizeof(float));
  cudaMalloc(&d_partial_ready, cluster_size * sizeof(int));
  cudaMemset(d_partial_ready, 0, cluster_size * sizeof(int));

  size_t smem_bytes = (TILE_M*TILE_K + TILE_K*TILE_N + rows_per_rank*TILE_N) * sizeof(float);

  int lda=K, ldb=N, ldc=N;
  switch (cluster_size) {
    case 2:  launch_cs<2 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile, d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 4:  launch_cs<4 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile, d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 8:  launch_cs<8 >(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile, d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
    case 16: launch_cs<16>(A,B,C, lda,ldb,ldc, d_ready,num_k_slices, d_A_tile,d_B_tile, d_C_partials,d_partial_ready,cluster_size, smem_bytes); break;
  }
  cudaDeviceSynchronize();

  float hC[M*N];
  cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n");
  for(int r=0;r<8;++r){ for(int c=0;c<8;++c) printf("%7.3f ", hC[r*N+c]); printf("\n"); }

  cudaFree(d_partial_ready); cudaFree(d_C_partials);
  cudaFree(d_B_tile); cudaFree(d_A_tile);
  cudaFree(d_ready); cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
