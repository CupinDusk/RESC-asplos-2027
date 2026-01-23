#!/bin/bash

# 获取 RESERVE_ELEMS 参数，默认为 1
RESERVE_ELEMS=${1:-1}
CLUSTER_NUMS=${2:-16}
# 创建日志目录

# 编译所有配置
nvcc -arch=sm_90 flash.cu -o cgcgresc${RESERVE_ELEMS} -DBACKEND=2 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DSTRIDE_ELEMS=16384 -DREAD_MODE=0 -DWRITE_MODE=0 -lcudart && patchelf --remove-rpath cgcgresc${RESERVE_ELEMS}

# 运行所有配置
./cgcgresc${RESERVE_ELEMS} 2 1 ${CLUSTER_NUMS} 512 > logtext/resc/${RESERVE_ELEMS}cgcg21_${CLUSTER_NUMS}.log
./cgcgresc${RESERVE_ELEMS} 4 1 ${CLUSTER_NUMS} 512 > logtext/resc/${RESERVE_ELEMS}cgcg41_${CLUSTER_NUMS}.log
./cgcgresc${RESERVE_ELEMS} 8 1 ${CLUSTER_NUMS} 512 > logtext/resc/${RESERVE_ELEMS}cgcg81_${CLUSTER_NUMS}.log