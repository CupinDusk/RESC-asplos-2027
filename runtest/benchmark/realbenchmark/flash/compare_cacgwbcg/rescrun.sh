#!/bin/bash

# 获取 RESERVE_ELEMS 参数，默认为 1
RESERVE_ELEMS=${1:-1}

# 创建日志目录

# 编译所有配置
nvcc -arch=sm_90 flash.cu -o cgcgresc${RESERVE_ELEMS} -DBACKEND=2 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DREAD_MODE=0 -DWRITE_MODE=0 -lcudart && patchelf --remove-rpath cgcgresc${RESERVE_ELEMS}
nvcc -arch=sm_90 flash.cu -o cgwbresc${RESERVE_ELEMS} -DBACKEND=2 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DREAD_MODE=0 -DWRITE_MODE=1 -lcudart && patchelf --remove-rpath cgwbresc${RESERVE_ELEMS}
nvcc -arch=sm_90 flash.cu -o cawbresc${RESERVE_ELEMS} -DBACKEND=2 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DREAD_MODE=1 -DWRITE_MODE=1 -lcudart && patchelf --remove-rpath cawbresc${RESERVE_ELEMS}
nvcc -arch=sm_90 flash.cu -o cacgresc${RESERVE_ELEMS} -DBACKEND=2 -DRESERVE_ELEMS=${RESERVE_ELEMS} -DREAD_MODE=1 -DWRITE_MODE=0 -lcudart && patchelf --remove-rpath cacgresc${RESERVE_ELEMS}

# 运行所有配置
./cacgresc${RESERVE_ELEMS} 2 1 > logtext/resc/${RESERVE_ELEMS}cacg21.log
./cawbresc${RESERVE_ELEMS} 2 1 > logtext/resc/${RESERVE_ELEMS}cawb21.log
./cgwbresc${RESERVE_ELEMS} 2 1 > logtext/resc/${RESERVE_ELEMS}cgwb21.log
./cgcgresc${RESERVE_ELEMS} 2 1 > logtext/resc/${RESERVE_ELEMS}cgcg21.log

./cacgresc${RESERVE_ELEMS} 4 1 > logtext/resc/${RESERVE_ELEMS}cacg41.log
./cawbresc${RESERVE_ELEMS} 4 1 > logtext/resc/${RESERVE_ELEMS}cawb41.log
./cgwbresc${RESERVE_ELEMS} 4 1 > logtext/resc/${RESERVE_ELEMS}cgwb41.log
./cgcgresc${RESERVE_ELEMS} 4 1 > logtext/resc/${RESERVE_ELEMS}cgcg41.log

./cacgresc${RESERVE_ELEMS} 8 1 > logtext/resc/${RESERVE_ELEMS}cacg81.log
./cawbresc${RESERVE_ELEMS} 8 1 > logtext/resc/${RESERVE_ELEMS}cawb81.log
./cgwbresc${RESERVE_ELEMS} 8 1 > logtext/resc/${RESERVE_ELEMS}cgwb81.log
./cgcgresc${RESERVE_ELEMS} 8 1 > logtext/resc/${RESERVE_ELEMS}cgcg81.log