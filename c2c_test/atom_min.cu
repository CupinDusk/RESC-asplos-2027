#include <cstdio>
#include <cuda_runtime_api.h>
__global__ void k(int* a){ if(threadIdx.x==0) atomicAdd(&a[0], 1); }
int main(){ int* d; cudaMalloc(&d,128); cudaMemset(d,0,128);
  k<<<8,64>>>(d); cudaDeviceSynchronize();
  int h; cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost);
  printf("ATOM_MIN result=%d (expect 8)\n", h); return 0; }
