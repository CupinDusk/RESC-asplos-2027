#!/bin/bash
#   ./flash <cluster_size:2|4|8> <repeat> <num_clusters> <actual_elems>
# 获取 RESERVE_ELEMS 参数，默认为 1
RESERVE_ELEMS=${1:-1}
CLUSTER_NUMS=${2:-16}

# 创建日志目录

# 编译所有配置
nvcc -arch=sm_90 flash.cu -o global${RESERVE_ELEMS} -DBACKEND=0 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DSTRIDE_ELEMS=16384 -lcudart && patchelf --remove-rpath global${RESERVE_ELEMS}

# 运行所有配置
./global${RESERVE_ELEMS} 2 1 ${CLUSTER_NUMS} 512 > logtext/global/${RESERVE_ELEMS}global21_${CLUSTER_NUMS}.log
./global${RESERVE_ELEMS} 4 1 ${CLUSTER_NUMS} 512 > logtext/global/${RESERVE_ELEMS}global41_${CLUSTER_NUMS}.log
./global${RESERVE_ELEMS} 8 1 ${CLUSTER_NUMS} 512 > logtext/global/${RESERVE_ELEMS}global81_${CLUSTER_NUMS}.log
