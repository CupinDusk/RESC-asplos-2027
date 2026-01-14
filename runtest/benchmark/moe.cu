#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// =========================================================
// 实验参数配置
// =========================================================
#define NUM_SMS 80             // 模拟 H100 的 SM 数量
#define THREADS_PER_BLOCK 256  // 每个 Block 的线程数
#define NUM_TOKENS 1000000     // 总 Token 数量
#define EXPERT_CAPACITY 4096   // 模拟：为了安全，DSMEM必须为每个Expert预留的空间

// =========================================================
// Helper: 错误检查
// =========================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// =========================================================
// Kernel 1: Baseline-Global / RESC (RESC Mode)
// ---------------------------------------------------------
// 说明：
// 这是 RESC 提倡的写法。数据直接写“逻辑上的全局地址”。
// 在 RESC 架构下，硬件会将其拦截并注入 L1。
// 在当前 GPU 上，这走 L2/HBM。
// 优势：不需要 Shared Memory，SM Occupancy 为 100%。
// =========================================================
__global__ void dynamic_routing_global_resc(
    const int* __restrict__ src_tokens,
    const int* __restrict__ dest_indices,
    int* __restrict__ global_output_buffer,
    int* __restrict__ global_counters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_TOKENS) return;

    int token_data = src_tokens[tid];
    int target_expert = dest_indices[tid];

    // 1. 争抢全局位置 (RESC下这将由L1原子操作加速)
    int offset = atomicAdd(&global_counters[target_expert], 1);

    // 2. 写入数据 (RESC下如果是簇内，这将是 WB 到 L1)
    // 这里的 stride 假设简单线性布局，实际场景可能更复杂
    global_output_buffer[target_expert * EXPERT_CAPACITY + offset] = token_data;
}

// =========================================================
// Kernel 2: Baseline-DSMEM (The Occupancy Killer)
// ---------------------------------------------------------
// 说明：
// 模拟 DSMEM 的困境。为了接收未知数量的数据，
// 我们必须声明巨大的 extern __shared__ memory。
// 这会导致 GPU 调度器无法在一个 SM 上启动多个 Block。
// =========================================================
__global__ void dynamic_routing_dsmem_simulation(
    const int* __restrict__ src_tokens,
    const int* __restrict__ dest_indices,
    int* __restrict__ global_output_buffer // 最终还是要写回，但经过 SMEM 暂存
) {
    // 【关键痛点】：必须静态分配巨大的 Shared Memory 作为接收 Buffer
    // 假设这个 Block 负责某个 Expert 的接收，或者作为中转
    // 在真实 DSMEM 代码中，这块内存属于 Target SM
    extern __shared__ int recv_buffer[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_TOKENS) return;

    // 模拟 DSMEM 写入过程：
    // 1. 写入 Shared Memory (极快)
    // 2. 但为此付出的代价是 Kernel 启动时的 shared_mem_config

    // 这里为了防止编译器优化掉，做一些假操作
    if (threadIdx.x < 10) {
        recv_buffer[threadIdx.x] = src_tokens[tid];
    }
}

// =========================================================
// 主函数：设置与运行
// =========================================================
int main() {
    int *d_src, *d_dest, *d_out, *d_count;
    int *h_dest;
    size_t tokens_size = NUM_TOKENS * sizeof(int);
    size_t out_size = NUM_SMS * EXPERT_CAPACITY * sizeof(int); // 巨大的输出空间

    // 1. 内存分配
    CUDA_CHECK(cudaMalloc(&d_src, tokens_size));
    CUDA_CHECK(cudaMalloc(&d_dest, tokens_size));
    CUDA_CHECK(cudaMalloc(&d_out, out_size));
    CUDA_CHECK(cudaMalloc(&d_count, NUM_SMS * sizeof(int)));

    h_dest = (int*)malloc(tokens_size);

    // 2. 初始化数据 (构造不规则负载)
    // 制造 Load Imbalance (Skewed Distribution)
    for (int i = 0; i < NUM_TOKENS; i++) {
        // 让 80% 的数据发给前 20% 的 SM (模拟热点)
        if (rand() % 100 < 80) {
            h_dest[i] = rand() % (NUM_SMS / 5);
        } else {
            h_dest[i] = rand() % NUM_SMS;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_dest, h_dest, tokens_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, NUM_SMS * sizeof(int)));

    // 计算 Grid 大小
    int num_blocks = (NUM_TOKENS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Running Experiment: Dynamic Routing (MoE/Graph Proxy)\n");
    printf("Total Tokens: %d, Target Experts: %d\n", NUM_TOKENS, NUM_SMS);

    // =========================================================
    // 实验 A: Global / RESC 模式
    // =========================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 只需要 0 Shared Memory -> 满 Occupancy
    dynamic_routing_global_resc<<<num_blocks, THREADS_PER_BLOCK, 0>>>(
        d_src, d_dest, d_out, d_count
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_global = 0;
    cudaEventElapsedTime(&ms_global, start, stop);

    // 计算并打印 Occupancy (理论值)
    int max_active_blocks_global;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_global, dynamic_routing_global_resc, THREADS_PER_BLOCK, 0);
    printf("\n[Mode 1: Global/RESC]\n");
    printf("  - Shared Mem Required: 0 KB\n");
    printf("  - Theoretical Max Active Blocks/SM: %d\n", max_active_blocks_global);
    printf("  - Latency: %.3f ms\n", ms_global);

    // =========================================================
    // 实验 B: DSMEM 模拟模式 (The Occupancy Wall)
    // =========================================================
    // 【核心实验逻辑】：
    // 为了防止 MoE 路由溢出，每个 Block 必须预留大量 Shared Memory。
    // 假设我们需要预留 64KB (H100 SMEM 228KB，但这通常被分割).
    // 即使我们不需要真的写进去，我们也必须申请这么大，才能模拟 DSMEM 的资源限制。
    int dsmem_reservation_size = 64 * 1024; // 64 KB per block

    cudaEventRecord(start);
    dynamic_routing_dsmem_simulation<<<num_blocks, THREADS_PER_BLOCK, dsmem_reservation_size>>>(
        d_src, d_dest, d_out
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_dsmem = 0;
    cudaEventElapsedTime(&ms_dsmem, start, stop);

    int max_active_blocks_dsmem;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_dsmem, dynamic_routing_dsmem_simulation, THREADS_PER_BLOCK, dsmem_reservation_size);

    printf("\n[Mode 2: Baseline DSMEM]\n");
    printf("  - Shared Mem Reserved (Worst-case Safety): %d KB\n", dsmem_reservation_size / 1024);
    printf("  - Theoretical Max Active Blocks/SM: %d  <-- LOOK AT THIS DROP!\n", max_active_blocks_dsmem);
    printf("  - Latency: %.3f ms\n", ms_dsmem);

    // 3. 验证结果 (对比 Occupancy)
    printf("\n[Conclusion]\n");
    printf("DSMEM forces a %.2fx drop in parallelism (Occupancy) due to static reservation.\n",
           (float)max_active_blocks_global / (float)max_active_blocks_dsmem);

    cudaFree(d_src); cudaFree(d_dest); cudaFree(d_out); cudaFree(d_count);
    free(h_dest);
    return 0;
}