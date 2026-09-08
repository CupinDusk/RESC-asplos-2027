// j1_probe.cu -- JOB 1 behavioral probe.
//
// Question: does ClusterSim invalidate the L1 at a Hopper cluster barrier
// (cluster.sync()), the way real H20 does via CCTL.IVALL?
//
// Method: hold the TOTAL number of unmarked-tile reads CONSTANT and vary only
// how many cluster barriers those reads are spread across.
//   A: EPOCHS=1, PASSES=P  -> 1 barrier
//   B: EPOCHS=P, PASSES=1  -> P barriers, IDENTICAL total accesses
// If the barrier invalidates L1, B must pay P sets of compulsory misses, so
// B_miss / A_miss ~= P.  If it does not, B_miss ~= A_miss.
//   C: same as B but with an explicit __threadfence() (membar.gl) before the
//      barrier.  The sim DOES wire membar to a whole-L1 invalidate when
//      -gpgpu_flush_l1_cache 1 (shader.cc:4069-4080). C is the POSITIVE CONTROL:
//      it proves the probe can detect an invalidate when one really happens.
//
// The tile is PRIVATE per block, read-only, never communicated, never marked --
// exactly the "unmarked compute tile" of the real-H20 E3 selectivity test.

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef EPOCHS
#define EPOCHS 1
#endif
#ifndef PASSES
#define PASSES 32
#endif
#ifndef TILE_KB
#define TILE_KB 16
#endif
#ifndef SYNCMODE
#define SYNCMODE 0   // 0=cluster.sync() only; 1=__threadfence()+cluster.sync(); 2=no sync
#endif
#ifndef NCLUSTERS
#define NCLUSTERS 2
#endif
#ifndef CS
#define CS 2
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p)); return v; }

__global__ __cluster_dims__(CS,1,1)
void probe(const float* __restrict__ tile, float* __restrict__ out, int tile_elems)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x;
  const int blk = blockIdx.x;
  const float* mytile = tile + (size_t)blk * (size_t)tile_elems;
  float acc = 0.f;
  for (int e = 0; e < EPOCHS; ++e) {
    for (int p = 0; p < PASSES; ++p) {
      for (int i = tid; i < tile_elems; i += blockDim.x)
        acc += ld_ca_f32(&mytile[i]);
    }
#if SYNCMODE == 1
    __threadfence();
#endif
#if SYNCMODE != 2
    cluster.sync();
#endif
  }
  if (tid == 0) out[blk] = acc;
#endif
}

int main(){
  const int tile_elems = (TILE_KB*1024)/(int)sizeof(float);
  const int nblocks = CS*NCLUSTERS;
  float *d_tile=nullptr,*d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_tile,(size_t)nblocks*tile_elems*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out,(size_t)nblocks*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_tile,0,(size_t)nblocks*tile_elems*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out,0,(size_t)nblocks*sizeof(float)));
  probe<<<nblocks,256>>>(d_tile,d_out,tile_elems);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float* h=(float*)malloc((size_t)nblocks*sizeof(float));
  CUDA_CHECK(cudaMemcpy(h,d_out,(size_t)nblocks*sizeof(float),cudaMemcpyDeviceToHost));
  double s=0; for(int i=0;i<nblocks;++i) s+=h[i];
  std::printf("J1PROBE EPOCHS=%d PASSES=%d TILE_KB=%d SYNCMODE=%d CS=%d nblocks=%d tile_elems=%d total_reads=%lld checksum=%.6f\n",
              EPOCHS,PASSES,TILE_KB,SYNCMODE,CS,nblocks,tile_elems,
              (long long)EPOCHS*(long long)PASSES*(long long)tile_elems*(long long)nblocks, s);
  free(h);
  return 0;
}
