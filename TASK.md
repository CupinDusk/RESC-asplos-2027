# 当前目标

将 `/legacy-research/owner/cudatest/RESC-c2c-wt` 的当前 RESC/ASPLOS 实验源码、实验说明和已有实验结果安全迁移到当前 GitHub 账号下的新私有仓库，使其他电脑可以从明确的 Git commit 继续构建、审计已有结果并继续实验。

# 任务需求说明

- 保留已被预注册和论文结果引用的历史锚点 `d3cc7ad7`。
- 保存 `gpu-cache.cc` MD5 为 `dfb94a326b42f43e9d3e2dfdd1f47e2e` 的当前未提交机制状态。
- GitHub 作为跨机器代码同步点；新仓库默认设为 private。
- 同步机制源码、benchmark 源码、实验配置、运行/提取脚本、工具链 shim、结果摘要、退出状态、统计文本和迁移说明。
- 原始 `*.log` 以确定性 `*.log.gz` 形式同步，并用 manifest 记录原文件大小和 SHA-256；不直接提交超过 GitHub 限制的未压缩日志。
- 不把构建产物、PTX/PTXAS/SASS、实验二进制、库、对象文件或备份文件加入新提交。

# 范围

## 包含

- 当前 Git 历史及 `d3cc7ad7` 锚点 tag。
- `src/` 中已跟踪的当前机制修改。
- `c2c_test/` 中可移植的源码、配置、脚本、`ptxas_shim` 和小型摘要。
- `c2c_test/` 中已运行实验的目录结构、退出码、统计文本、结果文件与压缩原始日志。
- 结果 manifest 和可重复生成压缩日志/manifest 的脚本。
- 当前 CUDA/编译器环境、关键 MD5、clone/build/验收说明。
- 新私有 GitHub 仓库、迁移分支和 tag。

## 不包含

- hLRC/VIPS-M backend 的设计与实现。
- 重跑或重新解释论文实验。
- 清理、删除或覆盖共享机器上的现有结果及未压缩原始日志。
- 改写、rebase、squash 或删除已有 Git 历史。
- 将未发表代码推到现有公开仓库 `CupinDusk/RESC`。

# 约束

- 不执行 `git add -A`。
- 不使用 broad `kill`、`pkill` 或递归删除。
- 不覆盖已有 `.gitignore`，只做增量补充。
- `d3cc7ad7` 必须在远端可达，不能只在本机保留 tag。
- 当前工具链基准包含 `/usr/local/cuda-12.4/bin/nvcc` 12.4.99、`clustersim-py311` 和 `ptxas_shim`。
- 所有待提交文件必须通过 staged 文件清单和体积审查。
- 新增 Git blob 不得超过 GitHub 100 MiB 单文件硬限制，不启用 Git LFS 或消耗额外配额。
- 压缩日志必须可解压校验，manifest 必须覆盖每个实验原始 `*.log`；`bin/` 中的编译日志不在范围内。

# 完成标准

- 新私有 GitHub 仓库已创建。
- 迁移分支已推送，远端 commit hash 与本地一致。
- `resc-published-anchor-d3cc7ad7` tag 已推送且指向 `d3cc7ad7`。
- 当前机制源码和必要实验配方已提交，编译生成物及未压缩原始日志未进入新提交。
- 已有实验的配置、说明、退出状态、统计文本、小型结果和确定性压缩日志已提交。
- manifest 的日志数量与本地符合范围的实验 `*.log` 数量一致，抽样及全量 gzip 完整性检查通过。
- 已确认新增文件中无 PTX/PTXAS/SASS、ELF 二进制、库、对象文件或超过 100 MiB 的 blob。
- 迁移说明记录关键 MD5、工具链、构建入口、验收控制点和已知缺失项。
- 从远端只读查询能够看到迁移分支和 tag。
