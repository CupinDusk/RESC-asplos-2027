// dcomb.cu -- D-path reducible epoch-scoped cluster COMBINING (pre-reg 2026-08-30).
// Q: does an endpoint log-depth REDUCTION TREE over the SM-to-SM fabric beat
// (a) correct-global atomics and (b) a FAIR batched single-owner FUNNEL, wall-clock?
// MAKE-OR-BREAK: do the per-round pairwise peer reads PARALLELIZE (round cost flat
// in N) or SERIALIZE (linear in N = the combiner-Gate2 recurrence)?
//
// LOCAL-REDUCE PHASE IDENTICAL ACROSS ALL 3 ARMS (bit-for-bit). Only COMBINE differs.
// Integer-only => bit-exact CPU reference.
//
// TRANSPORT (-DTRANSPORT): 0 = DSM smem map_shared_rank (proven 195cyc fabric route);
//   1 = RESC marked-global disjoint line via ld.global.ca/st.global.wb (RESC_MW path).
// CLUSTER_SIZE (-DCLUSTER_SIZE)=N ranks = separate 1-core sim clusters under cfg_h20
//   (n_cores_per_cluster=1) => peer reads traverse sm_2_sm (195cyc).
//
// mode = "global" | "funnel" | "tree".
//   global : each rank atomicAdds its K partials into SHARED cluster global bins.
//   funnel : owner rank0 reads all N-1 peer partials SERIALLY (combine depth N-1).
//   tree   : reduction tree, log2(N) rounds; round r: receivers (bit r == 0) read
//            partner (rank + 2^r) partial and add; 1 barrier/round. Depth ceil(log2 N).
//
// Critical path = rank 0 (deepest; epoch closes when slowest rank done). We record
// rank0 combine cycles AND per-round cycles (cluster 0) to test parallel-vs-serial.

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
#ifndef TRANSPORT
#define TRANSPORT 1
#endif
#ifndef MAXK
#define MAXK 256
#endif
#define LINE_INTS 32
#define PAD(K) ((((K)+LINE_INTS-1)/LINE_INTS)*LINE_INTS)

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

__device__ unsigned long long g_local_cyc[8192];   // per-block local-reduce cycles
__device__ unsigned long long g_comb_cyc[8192];     // per-block combine cycles
__device__ unsigned long long g_round_cyc[16];      // cluster0 rank0 per-round combine cycles
__device__ int                g_nrounds;

// RESC hook: mark a global address as a cluster-coherent line (functional in sim).
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){ asm volatile("" :: "l"(addr) : "memory"); }
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){ return x & ~((uintptr_t)128-1); }
__device__ __forceinline__ void register_range_warp0(void* base, size_t bytes){
  uintptr_t b=(uintptr_t)base, start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+127)/128);
  int lane=threadIdx.x&31;
  for(int off=0;off<nlines;off+=32){ int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*128;
    gpgpusim_register_cluster_coherent_addr((void*)line); }
}
__device__ __forceinline__ int  ld_ca_s32(const int* p){ int v; asm volatile("ld.global.ca.s32 %0,[%1];":"=r"(v):"l"(p)); return v; }
__device__ __forceinline__ void st_wb_s32(int* p,int v){ asm volatile("st.global.wb.s32 [%0],%1;"::"l"(p),"r"(v):"memory"); }

__host__ __device__ __forceinline__ int contrib(int cid,int rank,int b,int j){
  return (int)((cid*131 + rank*17 + b*7 + j*3 + 1) & 0xFF);
}

template<int CS>
__global__ void __cluster_dims__(CS,1,1)
comb_kernel(int* __restrict__ g_bins, int* __restrict__ g_part,
            int mode, int K, int LOCAL, int PADK){
#if __CUDA_ARCH__ >= 900
  __shared__ int s_part[MAXK];
  __shared__ int s_tmp[MAXK];
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int cid  = (int)(blockIdx.x / CS);
  const int lane = threadIdx.x & 31;
  int* my_gpart = g_part + (size_t)(cid*CS + rank)*PADK;
#if TRANSPORT==1
  if (lane<32) register_range_warp0(my_gpart, (size_t)K*sizeof(int));
#endif

  unsigned long long t_entry = clock64();
  // ---- LOCAL REDUCE (identical across all arms) ----
  for (int b=lane; b<K; b+=32){
    int acc=0;
    for (int j=0;j<LOCAL;++j) acc += contrib(cid,rank,b,j);
    s_part[b]=acc;
#if TRANSPORT==1
    st_wb_s32(&my_gpart[b], acc);
#endif
  }
  __syncwarp();
  cl.sync();                          // EPOCH boundary
  unsigned long long t0 = clock64();

  // ---- COMBINE (differs per arm) ----
  if (mode==0){                       // GLOBAL: atomic into shared cluster bins
    int* my_bins = g_bins + (size_t)cid*PADK;
    for (int b=lane; b<K; b+=32) atomicAdd(&my_bins[b], s_part[b]);
    cl.sync();
  } else if (mode==1){                // FUNNEL: owner reads N-1 peers SERIALLY
    if (rank==0){
      for (int p=1;p<CS;++p){
#if TRANSPORT==0
        int* pp = (int*)cl.map_shared_rank(s_part, p);
        for (int b=lane; b<K; b+=32) s_part[b] += pp[b];
#else
        int* pp = g_part + (size_t)(cid*CS + p)*PADK;
        for (int b=lane; b<K; b+=32) s_part[b] += ld_ca_s32(&pp[b]);
#endif
        __syncwarp();
      }
    }
    cl.sync();
  } else {                            // TREE: reduction tree, 1 barrier/round
    int roundi=0;
    for (int step=1; step<CS; step<<=1){
      // round cost = [before this round's reads .. all receivers done at barrier].
      // In round r there are N/2^(r+1) concurrent receiver->peer reads over the
      // fabric; if they serialize on a shared port, the barrier waits for the last
      // one => round cost grows with #concurrent readers (linear-in-N recurrence).
      // If they parallelize, round cost is ~flat (one read latency) across rounds.
      unsigned long long rt0=0;
      if (cid==0 && rank==0 && lane==0) rt0=clock64();
      if ((rank & (step-1))==0 && (rank & step)==0){   // receiver this round
        int peer = rank + step;
        if (peer < CS){
#if TRANSPORT==0
          int* pp = (int*)cl.map_shared_rank(s_part, peer);
          for (int b=lane; b<K; b+=32) s_tmp[b] = pp[b];
#else
          int* pp = g_part + (size_t)(cid*CS + peer)*PADK;
          for (int b=lane; b<K; b+=32) s_tmp[b] = ld_ca_s32(&pp[b]);
#endif
          __syncwarp();
          for (int b=lane; b<K; b+=32){
            s_part[b] += s_tmp[b];
#if TRANSPORT==1
            st_wb_s32(&my_gpart[b], s_part[b]);
#endif
          }
          __syncwarp();
        }
      }
      cl.sync();                      // barrier gates next round on this round's adds
      if (cid==0 && rank==0 && lane==0){ unsigned long long rt1=clock64();
        if (roundi<16) g_round_cyc[roundi]=(rt1>rt0)?(rt1-rt0):0ULL; }
      roundi++;
    }
    if (cid==0 && rank==0 && lane==0) g_nrounds=roundi;
  }
  unsigned long long t1 = clock64();

  if (lane==0){
    unsigned slot = blockIdx.x & 8191;
    g_local_cyc[slot] = (t0>t_entry)?(t0-t_entry):0ULL;
    g_comb_cyc[slot]  = (t1>t0)?(t1-t0):0ULL;
  }

  // final result -> shared cluster bins (rank0 holds full sum for funnel & tree)
  if (mode!=0 && rank==0){
    int* my_bins = g_bins + (size_t)cid*PADK;
    for (int b=lane; b<K; b+=32) my_bins[b] = s_part[b];
  }
  cl.sync();                          // residency (NOT timed)
#endif
}

int main(int argc,char**argv){
  const char* ms = (argc>1)?argv[1]:"tree";
  int K       = (argc>2)?atoi(argv[2]):32;
  int LOCAL   = (argc>3)?atoi(argv[3]):64;
  int nclust  = (argc>4)?atoi(argv[4]):8;
  int block   = (argc>5)?atoi(argv[5]):32;
  if(K<1)K=1; if(K>MAXK)K=MAXK; if(LOCAL<1)LOCAL=1; if(nclust<1)nclust=1; if(block<1)block=32;
  const int CS=CLUSTER_SIZE;
  const int PADK=PAD(K);
  int mode = (strcmp(ms,"global")==0)?0 : (strcmp(ms,"funnel")==0)?1 : 2;

  int *d_bins=nullptr,*d_part=nullptr;
  const size_t nbins=(size_t)nclust*PADK;
  const size_t npart=(size_t)nclust*CS*PADK;
  CUDA_CHECK(cudaMalloc(&d_bins,nbins*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_part,npart*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_bins,0,nbins*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_part,0,npart*sizeof(int)));

  dim3 grid(CS*nclust,1,1), blk(block,1,1);
  cudaFuncSetAttribute(comb_kernel<CLUSTER_SIZE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  comb_kernel<CLUSTER_SIZE><<<grid,blk>>>(d_bins,d_part,mode,K,LOCAL,PADK);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int* h_bins=(int*)malloc(nbins*sizeof(int));
  CUDA_CHECK(cudaMemcpy(h_bins,d_bins,nbins*sizeof(int),cudaMemcpyDeviceToHost));
  unsigned long long lc[8192], cc[8192], rr[16]; int nr=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(lc,g_local_cyc,sizeof(lc)));
  CUDA_CHECK(cudaMemcpyFromSymbol(cc,g_comb_cyc,sizeof(cc)));
  CUDA_CHECK(cudaMemcpyFromSymbol(rr,g_round_cyc,sizeof(rr)));
  cudaMemcpyFromSymbol(&nr,g_nrounds,sizeof(int));

  // bit-exact reference per cluster.
  int mism=0,first=-1; long long devsum=0,refsum=0;
  for(int c=0;c<nclust;++c) for(int b=0;b<K;++b){
    long long r=0; for(int rk=0;rk<CS;++rk) for(int j=0;j<LOCAL;++j) r+=contrib(c,rk,b,j);
    long long dv=h_bins[(size_t)c*PADK+b];
    devsum+=dv; refsum+=r;
    if(dv!=r){ mism++; if(first<0)first=c*PADK+b; }
  }
  int bit=(mism==0 && devsum==refsum)?1:0;

  // critical-path combine = rank0 (blockIdx%CS==0). local = same (any rank, symmetric).
  unsigned long long lv[8192], cv[8192]; int nl=0,nc=0;
  for(int bi=0; bi<CS*nclust; ++bi){
    int rk = bi % CS;
    if(rk==0){ if(cc[bi]>0) cv[nc++]=cc[bi]; if(lc[bi]>0) lv[nl++]=lc[bi]; }
  }
  auto med=[&](unsigned long long*a,int n)->double{ if(n==0)return 0.0;
    for(int i=1;i<n;++i){ auto k=a[i]; int j=i-1; while(j>=0&&a[j]>k){a[j+1]=a[j];--j;} a[j+1]=k; }
    return (double)a[n/2]; };
  double lmed=med(lv,nl), cmed=med(cv,nc);

  printf("DCOMB mode=%s N=%d TRANSPORT=%d K=%d LOCAL=%d nclust=%d PADK=%d "
         "rank0_local=%.0f rank0_comb=%.0f BITIDENTICAL=%d mism=%d first=%d devsum=%lld refsum=%lld",
         ms,CS,TRANSPORT,K,LOCAL,nclust,PADK, lmed,cmed, bit,mism,first, devsum,refsum);
  if(mode==2){ printf(" nrounds=%d round_cyc=[",nr);
    for(int i=0;i<nr && i<16;++i) printf("%s%llu",(i?",":""),rr[i]);
    printf("]"); }
  printf("\n");
  free(h_bins);
  CUDA_CHECK(cudaFree(d_bins)); CUDA_CHECK(cudaFree(d_part));
  return 0;
}
