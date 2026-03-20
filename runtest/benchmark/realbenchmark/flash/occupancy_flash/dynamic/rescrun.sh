#!/bin/bash
set -euo pipefail

# 用法:
#   ./run.sh [RESERVE_ELEMS] [NUM_CLUSTERS] [REPEAT] [AVG_ELEMS] [BURST_PERIOD] [REUSE_PASSES]
#
# 例如:
#   ./run.sh 8192 16 1 512 16 4

RESERVE_ELEMS=${1:-8192}
NUM_CLUSTERS=${2:-16}
REPEAT=${3:-1}
AVG_ELEMS=${4:-512}
BURST_PERIOD=${5:-16}
REUSE_PASSES=${6:-4}

SRC="resc.cu"

mkdir -p logtext/global
mkdir -p logtext/dsm
mkdir -p logtext/resc

build_bin () {
    local out=$1
    local backend=$2
    local extra_flags=$3

    nvcc -arch=sm_90 \
        "${SRC}" \
        -o "${out}" \
        -DBACKEND="${backend}" \
        -DRESERVE_ELEMS="${RESERVE_ELEMS}" \
        -DSTRIDE_ELEMS=16384 \
        ${extra_flags} \
        -lcudart

    patchelf --remove-rpath "${out}" || true
}

# 编译三种模式
#build_bin "global${RESERVE_ELEMS}" 0 ""
#build_bin "dsm${RESERVE_ELEMS}"    1 ""
build_bin "resc${RESERVE_ELEMS}"   2 "-DREAD_MODE=1 -DWRITE_MODE=1"

run_one () {
    local exe=$1
    local mode=$2
    local cs=$3

    ./"${exe}" \
        cluster_size="${cs}" \
        repeat="${REPEAT}" \
        num_clusters="${NUM_CLUSTERS}" \
        avg_elems="${AVG_ELEMS}" \
        burst_period="${BURST_PERIOD}" \
        reuse_passes="${REUSE_PASSES}" \
        > "logtext/${mode}/${RESERVE_ELEMS}_${mode}_c${cs}_n${NUM_CLUSTERS}_r${REPEAT}_a${AVG_ELEMS}_b${BURST_PERIOD}_u${REUSE_PASSES}.log"
}

for cs in 2 4 8; do
    #run_one "global${RESERVE_ELEMS}" "global" "${cs}"
    #run_one "dsm${RESERVE_ELEMS}"    "dsm"    "${cs}"
    run_one "resc${RESERVE_ELEMS}"   "resc"   "${cs}"
done