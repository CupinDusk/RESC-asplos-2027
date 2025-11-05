// nvcc -std=c++17 -arch=sm_90 -Xptxas -O0 -lineinfo cluster.cu -o cluster -lcudart
// 运行示例： ./cluster 8   # cluster_size 可传 2/4/8/16

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
#define TILE_K 32       // 两个 k-slice
#endif
static_assert(M%TILE_M==0 && N%TILE_N==0 && K%TILE_K==0, "dims must be multiples");

__host__ __device__ inline int idx2(int r,int c,int ld){ return r*ld + c; }
// 轻量 busy-wait（避免 PTX nanosleep）
__device__ inline void spin_until_one(volatile int* flag){
  while (atomicAdd((int*)flag, 0) == 0) {
    #pragma unroll 4
    for (int s = 0; s < 64; ++s) {
      asm volatile("");        // 小延时，避免过热自旋
    }
    __threadfence_block();     // 保持 block 内可见性
  }
}

template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void gemm_dsm_pull_pingpong_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int ldA, int ldB, int ldC,
                                   // GMEM 小同步变量（甜蜜区保留在 L2/GMEM）
                                   int* __restrict__ ready, int ready_len)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int rows_per_rank = TILE_M / CLUSTER_SIZE;

  // 动态 SMEM：A0/B0 + A1/B1（双缓冲），B 用 (TILE_N+1) 步长 padding，C_loc 为本 CTA 行段
  extern __shared__ float smem[];
  // A0/A1:  [TILE_M x TILE_K] 正常步长 TILE_K
  // B0/B1:  [TILE_K x (TILE_N+1)] 行步长 TILE_N+1（padding 降冲突，读写 B 索引时用 strideN = TILE_N+1）
  const int A_elems = TILE_M * TILE_K;                  // 2048
  const int B_elems = TILE_K * (TILE_N + 1);            // 32*(64+1)=2080
  float* A0 = smem;
  float* B0 = A0 + A_elems;
  float* A1 = B0 + B_elems;
  float* B1 = A1 + A_elems;
  float* C_loc = B1 + B_elems;                          // [rows_per_rank x TILE_N]

  const int row0 = 0, col0 = 0;
  const int my_row_start = rank * rows_per_rank;
  const int my_row_end   = my_row_start + rows_per_rank;
  const int num_k_slices = K / TILE_K;                  // 2
  const int strideN = TILE_N + 1;                       // B 的行步长（padding）

  // 清零本 CTA 的行段
  for (int lin = threadIdx.x; lin < rows_per_rank*TILE_N; lin += blockDim.x) C_loc[lin] = 0.f;
  __syncthreads();

  // ========== ks = 0 先预热装载 ==========
  {
    const int ks = 0;
    const int producer = ks % CLUSTER_SIZE;
    const bool i_am_producer = (rank == producer);
    const int  k_base = ks * TILE_K;

    // 生产者装到 A0/B0（B0 带 padding）
    if (i_am_producer) {
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x){
        int r=i/TILE_K, c=i%TILE_K; A0[i] = A[idx2(row0 + r, k_base + c, ldA)];
      }
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){
        int r=i/TILE_N, c=i%TILE_N; B0[r*strideN + c] = B[idx2(k_base + r, col0 + c, ldB)];
      }
      __threadfence_block();
      if (threadIdx.x==0) { ready[ks] = 1; __threadfence(); }
    }
    if (!i_am_producer && threadIdx.x==0) spin_until_one(&ready[ks]);
    __syncthreads();

    // 消费者远程→本地拷贝 A0/B0；不做 cluster.sync（用 flag+fence 保证可见性）
    if (!i_am_producer) {
      float* A_remote = cluster.map_shared_rank(A0, producer);
      float* B_remote = cluster.map_shared_rank(B0, producer);
      for (int i=threadIdx.x;i<A_elems;i+=blockDim.x) A0[i] = A_remote[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){ int r=i/TILE_N, c=i%TILE_N; B0[r*strideN + c] = B_remote[r*strideN + c]; }
      __syncthreads();
    }
  }

  // ========== 主循环：计算 ks，同步预取 ks+1 到另一套缓冲 ==========
  for (int ks=0; ks<num_k_slices; ++ks) {
    const int next = ks + 1;
    const int producer_next = next % CLUSTER_SIZE;
    const bool has_next = (next < num_k_slices);
    const bool i_am_producer_next = (rank == producer_next);
    const int  k_base_next = next * TILE_K;
    const int parity = ks & 1;

    // 生产者抢先装 ks+1 -> A1/B1 或 A0/B0（双缓冲），与 ks 的计算重叠
    if (has_next && i_am_producer_next) {
      float* A_dst = parity ? A0 : A1;
      float* B_dst = parity ? B0 : B1;
      for (int i=threadIdx.x;i<TILE_M*TILE_K;i+=blockDim.x){
        int r=i/TILE_K, c=i%TILE_K; A_dst[i] = A[idx2(row0 + r, k_base_next + c, ldA)];
      }
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){
        int r=i/TILE_N, c=i%TILE_N; B_dst[r*strideN + c] = B[idx2(k_base_next + r, col0 + c, ldB)];
      }
      __threadfence_block();
      if (threadIdx.x==0) { ready[next] = 1; __threadfence(); }
    }

    // 计算 ks：从本地（已就绪的 A/B 缓冲）读取
    float* A_loc = (parity ? A1 : A0);
    float* B_loc = (parity ? B1 : B0);
    for (int mi = my_row_start + threadIdx.x; mi < my_row_end; mi += blockDim.x) {
      for (int ni=0; ni<TILE_N; ++ni) {
        float acc = C_loc[(mi - my_row_start) * TILE_N + ni];
        #pragma unroll
        for (int kk=0; kk<TILE_K; ++kk) {
          float a = A_loc[mi*TILE_K + kk];
          float b = B_loc[kk*strideN + ni];
          acc = fmaf(a, b, acc);
        }
        C_loc[(mi - my_row_start) * TILE_N + ni] = acc;
      }
    }
    __syncthreads();

    // 消费者：并行拷贝 next（若存在）到本地的另一套缓冲（不阻塞当前计算）
    if (has_next && !i_am_producer_next) {
      if (threadIdx.x==0) spin_until_one(&ready[next]);
      __syncthreads();
      float* A_remote = cluster.map_shared_rank(parity ? A0 : A1, producer_next);
      float* B_remote = cluster.map_shared_rank(parity ? B0 : B1, producer_next);
      float* A_dst    = parity ? A0 : A1;
      float* B_dst    = parity ? B0 : B1;
      for (int i=threadIdx.x;i<A_elems;i+=blockDim.x) A_dst[i] = A_remote[i];
      for (int i=threadIdx.x;i<TILE_K*TILE_N;i+=blockDim.x){ int r=i/TILE_N, c=i%TILE_N; B_dst[r*strideN + c] = B_remote[r*strideN + c]; }
      __syncthreads();
    }
  }

  // 仅在最终聚合前做一次簇域屏障（可保留/可去：取决于你是否用 GMEM 小 flag）
  cluster.sync();

  // rank0 聚合：远程读 peers 的 C_loc 回写 GMEM（一次）
  if (rank == 0) {
    for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
      int r_off = lin / TILE_N, c_off = lin % TILE_N;
      C[idx2(row0 + r_off, col0 + c_off, ldC)] = C_loc[lin];
    }
    __syncthreads();
    for (int p=1; p<CLUSTER_SIZE; ++p) {
      float* C_peer = cluster.map_shared_rank(C_loc, p);
      int base_row = p * rows_per_rank;
      for (int lin=threadIdx.x; lin<rows_per_rank*TILE_N; lin+=blockDim.x) {
        int r_off = lin / TILE_N, c_off = lin % TILE_N;
        C[idx2(row0 + base_row + r_off, col0 + c_off, ldC)] = C_peer[lin];
      }
      __syncthreads();
    }
  }
  cluster.sync();
#endif
}

// -------- Host --------
static void init_mat(float* p,int R,int C,float s){
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) p[r*C+c]= s*((r+c)%7-3);
}
template<int CS>
void launch_cs(float* A,float* B,float* C,int lda,int ldb,int ldc,int* d_ready,int ready_len,size_t smem_bytes){
  dim3 grid(CS,1,1), block(256,1,1);
  gemm_dsm_pull_pingpong_kernel<CS><<<grid, block, smem_bytes>>>(A,B,C, lda,ldb,ldc, d_ready, ready_len);
}

int main(int argc, char** argv){
  int cluster_size = 8; if (argc>=2) cluster_size = atoi(argv[1]);
  if (!(cluster_size==2||cluster_size==4||cluster_size==8||cluster_size==16)) { printf("cluster_size must be 2/4/8/16\n"); return 1; }
  if (TILE_M % cluster_size) { printf("TILE_M (%d) must be divisible by cluster_size (%d)\n",TILE_M,cluster_size); return 1; }

  float *A,*B,*C;
  cudaMalloc(&A, M*K*sizeof(float));
  cudaMalloc(&B, K*N*sizeof(float));
  cudaMalloc(&C, M*N*sizeof(float));
  { float *hA=(float*)malloc(M*K*sizeof(float)), *hB=(float*)malloc(K*N*sizeof(float));
    init_mat(hA, M, K, 0.1f); init_mat(hB, K, N, 0.2f);
    cudaMemcpy(A,hA,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B,hB,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(C,0,M*N*sizeof(float)); free(hA); free(hB); }

  const int num_k_slices = K / TILE_K;  // 2
  int* d_ready=nullptr; cudaMalloc(&d_ready, num_k_slices*sizeof(int)); cudaMemset(d_ready,0,num_k_slices*sizeof(int));

  // 动态 shared：A0/B0 + A1/B1 + C_loc
  const int rows_per_rank = TILE_M / cluster_size;
  size_t smem_bytes = (TILE_M*TILE_K*2 + (TILE_K*(TILE_N+1))*2 + rows_per_rank*TILE_N) * sizeof(float);

  int lda=K, ldb=N, ldc=N;
  switch (cluster_size) {
    case 2:  launch_cs<2 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 4:  launch_cs<4 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 8:  launch_cs<8 >(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
    case 16: launch_cs<16>(A,B,C, lda,ldb,ldc, d_ready, num_k_slices, smem_bytes); break;
  }
  cudaDeviceSynchronize();

  float hC[M*N]; cudaMemcpy(hC, C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0:8,0:8]:\n"); for(int r=0;r<8;++r){ for(int c=0;c<8;++c) printf("%7.3f ", hC[r*N+c]); printf("\n"); }

  cudaFree(d_ready); cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
