#!/bin/bash

# 获取 RESERVE_ELEMS 参数，默认为 1
RESERVE_ELEMS=${1:-1}

# 创建日志目录

# 编译所有配置
nvcc -arch=sm_90 flash.cu -o dsm${RESERVE_ELEMS} -DBACKEND=1 -DRESERVE_ELEMS=${RESERVE_ELEMS} -lcudart && patchelf --remove-rpath dsm${RESERVE_ELEMS}

# 运行所有配置
./dsm${RESERVE_ELEMS} 2 1 > logtext/dsm/${RESERVE_ELEMS}dsm21.log
./dsm${RESERVE_ELEMS} 4 1 > logtext/dsm/${RESERVE_ELEMS}dsm41.log
./dsm${RESERVE_ELEMS} 8 1 > logtext/dsm/${RESERVE_ELEMS}dsm81.log
