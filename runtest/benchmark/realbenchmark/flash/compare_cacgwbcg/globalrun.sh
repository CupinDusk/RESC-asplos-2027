#!/bin/bash

# 获取 RESERVE_ELEMS 参数，默认为 1
RESERVE_ELEMS=${1:-1}

# 创建日志目录

# 编译所有配置
nvcc -arch=sm_90 flash.cu -o global${RESERVE_ELEMS} -DBACKEND=0 -DRESERVE_ELEMS=${RESERVE_ELEMS} -lcudart && patchelf --remove-rpath global${RESERVE_ELEMS}

# 运行所有配置
./global${RESERVE_ELEMS} 2 1 > logtext/global/${RESERVE_ELEMS}global21.log
./global${RESERVE_ELEMS} 4 1 > logtext/global/${RESERVE_ELEMS}global41.log
./global${RESERVE_ELEMS} 8 1 > logtext/global/${RESERVE_ELEMS}global81.log
