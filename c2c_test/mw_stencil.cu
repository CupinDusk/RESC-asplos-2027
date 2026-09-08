// RESC MULTI-WRITER -- SECONDARY workload: integer Jacobi stencil with
// deliberate line-granular FALSE SHARING at partition boundaries.
//
// Domain of N cells partitioned across C blocks of a cluster. Block b owns cells
// [b*seg, (b+1)*seg). We lay the per-block boundary cells of block b and block
// b+1 ADJACENT in the SAME 128B line but DISTINCT words, so each cluster.sync
// epoch has two neighbors writing different words of one shared line = false
// sharing (path A, non-commutative -> per-word merge REQUIRED; whole-line LWW
// would clobber a neighbor's word if the sim merged cache data -- here the
// functional value is L2 so bit-identity holds, and the counters show per-word
// merges firing).
//
// INTEGER arithmetic only (bit-exact reference). K iterations.
//
// Arm selection: env RESC_MW (0=arm3, 1=arm2), RESC_MW_LWW=1 (arm4). Arm1 =
// -DRESC_MW_BASELINE_BUILD (no register hook).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#ifndef SEG
#define SEG 4                     // cells per block segment
#endif
#ifndef KITERS
#define KITERS 20
#endif

// Shared boundary line per cluster: one int per rank's right-boundary halo, all
// packed into ONE 128B line (16 ints). Neighbors write distinct words.
#define HALO_WORDS 32             // >= CLUSTER_SIZE, one 128B line = 32 int32

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  asm volatile("" :: "l"(addr) : "memory");
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)128 - 1);
}
__device__ __forceinline__ void register_range_warp0(void* base, size_t bytes){
  uintptr_t b=(uintptr_t)base, start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+127)/128);
  int lane=threadIdx.x&31;
  for(int off=0;off<nlines;off+=32){
    int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*128;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}
__device__ __forceinline__ int ld_ca_s32(const int* p){ int v;
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_wb_s32(int* p,int v){
  asm volatile("st.global.wb.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory"); }

__host__ __device__ __forceinline__ int init_cell(int cid,int gcell){
  return ((cid*7 + gcell*3 + 1) & 0xFF);
}

template<int CS>
__global__ void __cluster_dims__(CS,1,1)
stencil_kernel(int* __restrict__ halo, int num_clusters, int iters){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  int* my_halo = halo + (size_t)cid * HALO_WORDS;   // shared boundary line

#if !defined(RESC_MW_BASELINE_BUILD)
  if (rank == 0 && tid < 32)
    register_range_warp0(my_halo, (size_t)HALO_WORDS*sizeof(int));
#endif

  // each rank keeps its SEG cells in registers/local (private).
  int cells[SEG];
  int base_g = rank*SEG;
  for (int i=0;i<SEG;++i) cells[i] = init_cell(cid, base_g+i);

  // publish initial right-boundary into the shared line word `rank`.
  if (tid==0) st_wb_s32(&my_halo[rank], cells[SEG-1]);
  cl.sync();

  for (int it=0; it<iters; ++it){
    // Jacobi read-old / write-new with TWO syncs so the neighbor-word read is
    // deterministic (reads the fully-written previous-epoch line), while the
    // writes of distinct words to the shared line still race for false sharing.
    int left = (rank>0 && tid==0) ? ld_ca_s32(&my_halo[rank-1]) : 0;  // read OLD
    int newc0 = 0;
    if (tid==0){
      newc0 = (left + cells[0] + 1);              // depends on neighbor word
    }
    cl.sync();                                     // all reads done before writes
    if (tid==0){
      for (int i=SEG-1;i>0;--i) cells[i] = (cells[i]+cells[i-1]+1);
      cells[0] = newc0;
      st_wb_s32(&my_halo[rank], cells[SEG-1]);     // write NEW word (false sharing)
    }
    cl.sync();                                     // all writes done before next epoch
  }

  // final: write each rank's last cell into its halo word (already there).
#endif
}

// bit-exact CPU reference: replicate the exact same integer recurrence.
static void cpu_ref(int cid, int iters, int* out_halo /*HALO_WORDS*/){
  int cells[CLUSTER_SIZE][SEG];
  for (int rank=0; rank<CLUSTER_SIZE; ++rank)
    for (int i=0;i<SEG;++i) cells[rank][i]=init_cell(cid, rank*SEG+i);
  int halo[CLUSTER_SIZE];
  for (int rank=0; rank<CLUSTER_SIZE; ++rank) halo[rank]=cells[rank][SEG-1];
  for (int it=0; it<iters; ++it){
    int newhalo[CLUSTER_SIZE];
    int newc0[CLUSTER_SIZE];
    for (int rank=0; rank<CLUSTER_SIZE; ++rank){
      int left = (rank>0)? halo[rank-1] : 0;   // previous-epoch neighbor word
      newc0[rank] = left + cells[rank][0] + 1;
    }
    for (int rank=0; rank<CLUSTER_SIZE; ++rank){
      for (int i=SEG-1;i>0;--i) cells[rank][i] = cells[rank][i]+cells[rank][i-1]+1;
      cells[rank][0]=newc0[rank];
      newhalo[rank]=cells[rank][SEG-1];
    }
    for (int rank=0; rank<CLUSTER_SIZE; ++rank) halo[rank]=newhalo[rank];
  }
  for (int w=0; w<HALO_WORDS; ++w) out_halo[w]=0;
  for (int rank=0; rank<CLUSTER_SIZE; ++rank) out_halo[rank]=halo[rank];
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 64;
  int iters        = (argc>2)? atoi(argv[2]) : KITERS;
  if (num_clusters<1) num_clusters=1;
  if (iters<1) iters=1;

  const int CS=CLUSTER_SIZE;
  const size_t nwords=(size_t)num_clusters*HALO_WORDS;
  int* d_halo=nullptr;
  CUDA_CHECK(cudaMalloc(&d_halo, nwords*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_halo, 0, nwords*sizeof(int)));

  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  stencil_kernel<CLUSTER_SIZE><<<grid,block>>>(d_halo, num_clusters, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int* h_halo=(int*)malloc(nwords*sizeof(int));
  CUDA_CHECK(cudaMemcpy(h_halo, d_halo, nwords*sizeof(int), cudaMemcpyDeviceToHost));

  long long dev_checksum=0, ref_checksum=0; int mism=0, first_mis=-1;
  int* ref=(int*)malloc(HALO_WORDS*sizeof(int));
  for (int cid=0; cid<num_clusters; ++cid){
    cpu_ref(cid, iters, ref);
    for (int w=0; w<HALO_WORDS; ++w){
      int dv=h_halo[(size_t)cid*HALO_WORDS+w];
      dev_checksum+=dv; ref_checksum+=ref[w];
      if (dv!=ref[w]){ mism++; if(first_mis<0) first_mis=cid*HALO_WORDS+w; }
    }
  }
  printf("MW_STENCIL cluster=%d num_clusters=%d iters=%d SEG=%d "
         "OUT_CHECKSUM=%lld REF_CHECKSUM=%lld BITIDENTICAL=%d mismatches=%d first_mis=%d\n",
         CS, num_clusters, iters, SEG,
         dev_checksum, ref_checksum, (dev_checksum==ref_checksum && mism==0)?1:0,
         mism, first_mis);

  free(ref); free(h_halo);
  CUDA_CHECK(cudaFree(d_halo));
  return 0;
}
