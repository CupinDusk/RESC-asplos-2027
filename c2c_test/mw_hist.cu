// mw_hist.cu -- RESC multi-writer SKEWED ATOMIC-SCATTER histogram (pre-reg r18).
// Injection-bound (NOT HBM-bound) contention microbench. 8 SMs/cluster scatter
// small (4B) atomic adds into CLUSTER-SHARED bins (NOT per-block-private).
//
// THREE ARMS, byte-identical final histogram (same host-generated key stream):
//   arm0  (global)  : plain atom.global into GLOBAL bins -> every RMW serializes
//                     at L2. Build hist_mw_base (-DRESC_MW_BASELINE_BUILD, NO hook).
//   arm3  (RESC MW) : atom.global into RESC-MARKED cluster-coherent GLOBAL bins;
//                     RESC_MW=1 -> migrating owner-token resolves the RMW on the
//                     SM-to-SM fabric (1 hop / ownership change). Build hist_mw (hook).
//   arm-dsm         : per-rank SMEM bins; single-owner-per-bin (contiguous
//                     partition); each peer funnels EVERY update to the bin's owner
//                     via map_shared_rank + remote atomicAdd -> owner injection
//                     port drains all N updates = O(N) serialization (real DSM idiom
//                     under the no-privatization / immediate-visibility constraint).
//   mode selected by argv[1] = "global" | "dsm".
//
// Keys are host-generated (uniform / zipf1.0 / zipf1.5), deterministic, shared per
// rank across all clusters => the final per-bin histogram is identical for every
// arm by construction (bit-identity is a correctness precondition, not the result).
//
// 32-bit atomics only (sim's supported width; int64 hits a pre-existing coalescer
// 8-byte byte-mask bug, unrelated to RESC -- see mw_reduce.cu).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#define MAXK 256
#define LINE_INTS 32                 // 128B line = 32 int32

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// -------- sim hook: mark a global address as a cluster-coherent RESC line -------
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
#if defined(RESC_MW_BASELINE_BUILD)
  (void)addr;                                   // arm0: no marking at all
#else
  asm volatile("" :: "l"(addr) : "memory");     // intercepted functionally by sim
#endif
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

__device__ __forceinline__ int owner_of(int b, int K){
  int o = (int)(((long)b * CLUSTER_SIZE) / (K>0?K:1));   // contiguous partition
  if (o >= CLUSTER_SIZE) o = CLUSTER_SIZE-1;
  if (o < 0) o = 0;
  return o;
}

// ============================ arm0 / arm3 : GLOBAL atomics =====================
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
scatter_global(int* __restrict__ g_bins, const int* __restrict__ keys,
               int K, int UPD, int stride, int* __restrict__ sink){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int cid  = (int)(blockIdx.x / CS);
  const int tid  = threadIdx.x;
  int* my_bins = g_bins + (size_t)cid * stride;
#if !defined(RESC_MW_BASELINE_BUILD)
  if (rank==0 && tid<32) register_range_warp0(my_bins, (size_t)K*sizeof(int));
#endif
  cl.sync();
  const int* my_keys = keys + (size_t)rank * UPD;
  int prior = 0;                                   // CONSUME return -> forces atom.global
  for (int i=tid; i<UPD; i+=blockDim.x){
    int b = my_keys[i];
    prior += atomicAdd(&my_bins[b], 1);
  }
  cl.sync();
  if (tid==0) sink[blockIdx.x] = prior;            // keep the atom result live
#endif
}

// ============================ arm-dsm : single-owner SMEM funnel ===============
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
scatter_dsm(int* __restrict__ g_bins, const int* __restrict__ keys,
            int K, int UPD, int stride){
#if __CUDA_ARCH__ >= 900
  __shared__ int s_bins[MAXK];
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int cid  = (int)(blockIdx.x / CS);
  const int tid  = threadIdx.x;
  for (int b=tid; b<K; b+=blockDim.x) s_bins[b]=0;
  __syncthreads();
  cl.sync();
  const int* my_keys = keys + (size_t)rank * UPD;
  for (int i=tid; i<UPD; i+=blockDim.x){
    int b = my_keys[i];
    int o = owner_of(b, K);
    int* owner_bins = (o==rank) ? s_bins : (int*)cl.map_shared_rank(s_bins, o);
    atomicAdd(&owner_bins[b], 1);                  // remote if o!=rank -> DSM funnel
  }
  cl.sync();                                       // all remote updates delivered
  int* my_gbins = g_bins + (size_t)cid * stride;
  for (int b=tid; b<K; b+=blockDim.x)
    if (owner_of(b,K)==rank) my_gbins[b] = s_bins[b];   // owner publishes its bins
  cl.sync();
#endif
}

// ------------------------------ host key generation ---------------------------
static inline uint64_t splitmix64(uint64_t &s){
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int main(int argc, char** argv){
  const char* mode = (argc>1)? argv[1] : "global";      // global | dsm
  int K            = (argc>2)? atoi(argv[2]) : 64;
  int skew         = (argc>3)? atoi(argv[3]) : 2;        // 0=uniform 1=zipf1.0 2=zipf1.5
  int UPD          = (argc>4)? atoi(argv[4]) : 512;      // updates per rank
  int nclusters    = (argc>5)? atoi(argv[5]) : 8;
  int block        = (argc>6)? atoi(argv[6]) : 64;
  if (K<1) K=1; if (K>MAXK) K=MAXK;
  if (UPD<1) UPD=1; if (nclusters<1) nclusters=1; if (block<1) block=64;
  const int CS = CLUSTER_SIZE;
  const int stride = ((K + LINE_INTS-1)/LINE_INTS) * LINE_INTS;   // pad per-cluster to 128B

  double s = (skew==0)?0.0 : (skew==1)?1.0 : 1.5;

  // build zipf cdf over bins [0,K): weight(b) = 1/(b+1)^s ; bin 0 = hottest.
  double* w = (double*)malloc(K*sizeof(double));
  double tot=0; for(int b=0;b<K;b++){ w[b]=1.0/pow((double)(b+1), s); tot+=w[b]; }
  double* cdf=(double*)malloc(K*sizeof(double));
  double acc=0; for(int b=0;b<K;b++){ acc+=w[b]/tot; cdf[b]=acc; }

  // per-rank deterministic key streams, shared by all clusters.
  const size_t nkeys = (size_t)CS * UPD;
  int* h_keys = (int*)malloc(nkeys*sizeof(int));
  for (int rank=0; rank<CS; ++rank){
    uint64_t st = 0xD1B54A32D192ED03ULL ^ (0x100000001B3ULL*(uint64_t)(rank+1));
    for (int i=0;i<UPD;++i){
      uint64_t r = splitmix64(st);
      double u = (double)(r >> 11) / (double)(1ULL<<53);   // [0,1)
      // binary search cdf
      int lo=0, hi=K-1, b=K-1;
      while(lo<=hi){ int mid=(lo+hi)/2; if(u<=cdf[mid]){ b=mid; hi=mid-1;} else lo=mid+1; }
      h_keys[(size_t)rank*UPD + i] = b;
    }
  }

  // host reference histogram for ONE cluster (all clusters identical).
  long long* ref1 = (long long*)calloc(K, sizeof(long long));
  for (size_t i=0;i<nkeys;++i) ref1[h_keys[i]]++;
  long long ref_total_1c=0; for(int b=0;b<K;b++) ref_total_1c+=ref1[b];

  int *d_keys=nullptr, *d_bins=nullptr, *d_sink=nullptr;
  const size_t nbins = (size_t)nclusters * stride;
  CUDA_CHECK(cudaMalloc(&d_keys, nkeys*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bins, nbins*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sink, (size_t)nclusters*CS*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, nkeys*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bins, 0, nbins*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_sink, 0, (size_t)nclusters*CS*sizeof(int)));

  dim3 grid(CS*nclusters,1,1), blk(block,1,1);
  cudaFuncSetAttribute(scatter_global<CLUSTER_SIZE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(scatter_dsm<CLUSTER_SIZE>,    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(cudaDeviceSynchronize());

  int is_dsm = (strcmp(mode,"dsm")==0);
  if (is_dsm) scatter_dsm<CLUSTER_SIZE><<<grid,blk>>>(d_bins, d_keys, K, UPD, stride);
  else        scatter_global<CLUSTER_SIZE><<<grid,blk>>>(d_bins, d_keys, K, UPD, stride, d_sink);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int* h_bins=(int*)malloc(nbins*sizeof(int));
  CUDA_CHECK(cudaMemcpy(h_bins, d_bins, nbins*sizeof(int), cudaMemcpyDeviceToHost));

  // correctness: every cluster's bins must equal ref1[]; total = nclusters*ref_total_1c.
  long long dev_total=0; int mism=0, first_mis=-1;
  for (int c=0;c<nclusters;++c){
    for (int b=0;b<K;++b){
      long long dv = h_bins[(size_t)c*stride + b];
      dev_total += dv;
      if (dv != ref1[b]){ mism++; if(first_mis<0) first_mis=c*stride+b; }
    }
  }
  long long ref_total = (long long)nclusters * ref_total_1c;
  int bit = (mism==0 && dev_total==ref_total) ? 1 : 0;

  printf("MW_HIST mode=%s K=%d skew=%.1f UPD=%d nclusters=%d block=%d CS=%d "
         "hot0=%lld frac0=%.3f OUT_TOTAL=%lld REF_TOTAL=%lld BITIDENTICAL=%d mismatches=%d first_mis=%d\n",
         mode, K, s, UPD, nclusters, block, CS,
         ref1[0], (double)ref1[0]/(double)ref_total_1c,
         dev_total, ref_total, bit, mism, first_mis);

  free(w); free(cdf); free(h_keys); free(ref1); free(h_bins);
  CUDA_CHECK(cudaFree(d_keys)); CUDA_CHECK(cudaFree(d_bins)); CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
