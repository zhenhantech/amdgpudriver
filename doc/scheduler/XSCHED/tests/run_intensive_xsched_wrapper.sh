#!/bin/bash
# 包装脚本：清除环境变量后运行测试
# 用法（在 Docker 内）: bash /data/dockercode/run_intensive_xsched_wrapper.sh

# 保存当前目录
ORIG_DIR=$(pwd)

# 清除可能干扰的环境变量
unset LD_PRELOAD

# 运行测试脚本
cd /data/dockercode
bash /data/dockercode/run_intensive_xsched_only.sh

# 返回原目录
cd "$ORIG_DIR"
