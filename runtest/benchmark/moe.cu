#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;

// =========================================================
// 实验配置
// =========================================================
#define CLUSTER_SIZE 8          // H100 典型的 Cluster 大小
#define THREADS_PER_BLOCK 256   // 每个 Block 的线程数
#define NUM_TOKENS 10         // 每个线程处理的数据量

// 宏：检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// =========================================================
// Kernel A: 真实的 Hopper DSMEM 通信
// ---------------------------------------------------------
// 必须在编译时或运行时指定 Cluster 维度
// =========================================================
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
//__launch_bounds__(THREADS_PER_BLOCK, 8)
dsmem_moe_kernel(
    int* src_data,
    int* dst_dump,
    int buffer_capacity_ints // 动态传入缓冲区大小，用于检测溢出
) {
    // 1. 获取 Cluster 句柄
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    unsigned int tid = threadIdx.x;

    // 2. 定义动态 Shared Memory
    // 布局: [Counter (1 int)] + [Data Buffer (N ints)]
    extern __shared__ int smem_pool[];
    int* my_counter = &smem_pool[0];
    int* my_buffer  = &smem_pool[1];

    // 初始化本地计数器
    if (tid == 0) *my_counter = 0;

    // 3. 【核心 API】建立 DSMEM 映射表
    // 获取 Cluster 内其他所有 SM 的 Shared Memory 指针
    int* remote_counters[CLUSTER_SIZE];
    int* remote_buffers[CLUSTER_SIZE];

    // 这里的指针是 Generic Pointer，硬件会自动路由到对应的 SM
    #pragma unroll
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        remote_counters[i] = cluster.map_shared_rank(my_counter, i);
        remote_buffers[i]  = cluster.map_shared_rank(my_buffer, i);
    }

    // 4. 【核心 API】Cluster 同步
    // 确保所有 SM 都完成了 map 并且 smem 已分配
    cluster.sync();

    // 5. 模拟 MoE 动态路由 (Dynamic Routing)
    // 简单的伪随机生成器
    curandState state;
    curand_init(1234, rank * blockDim.x + tid, 0, &state);

    for (int i = 0; i < NUM_TOKENS; i++) {
        // 随机决定发给 Cluster 内的哪个 Expert (SM)
        int target_rank = curand(&state) % CLUSTER_SIZE;
        int data = src_data[tid]; // 模拟读取 Token

        // --- DSMEM 通信动作 ---

        // Step 1: 原子申请位置 (Remote Atomic Add)
        // 这一步走的是 Cluster Interconnect，不经过 L2
        int slot = atomicAdd(remote_counters[target_rank], 1);

        // Step 2: 写入数据 (Remote Store)
        // 【痛点验证】：必须检查边界！
        // 如果这里没有足够的 Shared Memory，数据就会丢失，或者需要回退到 Global Memory
        if (slot < buffer_capacity_ints) {
            remote_buffers[target_rank][slot] = data;
        }
        // else { 统计丢包或溢出 }
    }

    // 6. 结束同步
    cluster.sync();

    // (可选) 验证写入情况
    if (tid == 0 && rank == 0) {
        // printf("Rank 0 received %d tokens via DSMEM.\n", *my_counter);
    }
}

// =========================================================
// Kernel B: Global Memory (代表 RESC 的理想情况)
// ---------------------------------------------------------
// 不需要 Cluster API，不需要 Shared Memory 预留
// =========================================================
__global__ void global_moe_kernel(
    int* src_data,
    int* global_counters,
    int* global_buffer,
    int buffer_capacity_ints
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 伪随机 (保持逻辑一致)
    curandState state;
    curand_init(1234, tid, 0, &state);

    for (int i = 0; i < NUM_TOKENS; i++) {
        // 假设这里也是发给某个 Rank，但我们通过 Global Address 访问它
        int target_rank = curand(&state) % CLUSTER_SIZE;

        // 模拟 RESC：直接写 Global Address，硬件负责 L1 截获
        // 在现有硬件上，这走 L2，慢，但 Occupancy 高
        int slot = atomicAdd(&global_counters[target_rank], 1);

        // 全局内存非常大，几乎没有 Capacity 焦虑
        // 这里只是为了代码对称
        global_buffer[target_rank * buffer_capacity_ints + slot] = src_data[tid];
    }
}

// =========================================================
// Host 主函数
// =========================================================
int main(int argc, char** argv) {
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    if (prop.major < 9) {
        printf("Error: This code requires NVIDIA Hopper (SM90) architecture.\n");
        return -1;
    }

    // 实验参数
    // 模拟 MoE 场景：假设每个 Expert 需要接收很多 Token
    // 为了安全，我们必须预留 64KB 的 Shared Memory (H100 每个 SM 最大 228KB)
    int smem_reservation_bytes = 40 * 1024;
    int buffer_capacity_ints = (smem_reservation_bytes - sizeof(int)) / sizeof(int);

    // 计算 Grid
    int num_clusters = 10; // 启动 10 个 Cluster
    int num_blocks = num_clusters * CLUSTER_SIZE;

    // 准备数据
    int* d_src, *d_dump, *d_global_counters, *d_global_buffer;
    cudaMalloc(&d_src, num_blocks * THREADS_PER_BLOCK * sizeof(int));
    cudaMalloc(&d_dump, num_blocks * sizeof(int)); // Dummy output
    cudaMalloc(&d_global_counters, CLUSTER_SIZE * sizeof(int));
    // Global Buffer 很大，不用担心溢出
    cudaMalloc(&d_global_buffer, CLUSTER_SIZE * 1024 * 1024 * sizeof(int));

    printf("========================================================\n");
    printf("  Hopper DSMEM vs. Global/RESC Occupancy Analysis\n");
    printf("========================================================\n");
    printf("Scenario: MoE Dynamic Routing (Random Shuffle)\n");
    printf("Cluster Size: %d, Reservation Size: %d KB per Block\n", CLUSTER_SIZE, smem_reservation_bytes/1024);

    // ---------------------------------------------------------
    // 1. 运行 DSMEM Kernel (受限于 Shared Memory)
    // ---------------------------------------------------------

    // 设置 Kernel 的 Shared Memory 属性，允许使用大页
    //cudaFuncSetAttribute(dsmem_moe_kernel,
    //                     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                     smem_reservation_bytes);

    //

    // 初始化（否则 atomicAdd 的 counters 是未定义值）
    CUDA_CHECK(cudaMemset(d_global_counters, 0, CLUSTER_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dump, 0, num_blocks * sizeof(int)));

    // 关键：真正“占用多少动态SMEM”发生在 launch 的第三个参数
    // 这里传 smem_reservation_bytes，才会让每个block动态SMEM=40KB
    dsmem_moe_kernel<<<num_blocks, THREADS_PER_BLOCK, smem_reservation_bytes>>>(
        d_src, d_dump, buffer_capacity_ints
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n[Mode 1: DSMEM (Hopper Native)]\n");
    printf("  - Need map_shared_rank? YES\n");
    printf("  - Need cluster.sync?    YES\n");
    printf("  - Shared Memory Used:   %d KB\n", smem_reservation_bytes/1024);
    //printf("  ---> Max Active Blocks / SM: %d (The Occupancy Wall)\n", dsmem_max_blocks);

    // ---------------------------------------------------------
    // 2. 运行 Global/RESC Kernel (无 Shared Memory 负担)
    // ---------------------------------------------------------


    // 清理
    cudaFree(d_src); cudaFree(d_dump);
    cudaFree(d_global_counters); cudaFree(d_global_buffer);
    return 0;
}