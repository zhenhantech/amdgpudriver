#!/bin/bash
# 应用 XSched 修复并重新编译
set -e

echo "========================================================================"
echo "XSched 修复：支持默认流"
echo "========================================================================"
echo ""

XSCHED_SRC="/data/dockercode/xsched-official"
XSCHED_BUILD="/data/dockercode/xsched-build"
CONTAINER="zhenflashinfer_v1"

echo "[1/5] 备份原始文件..."
docker exec "$CONTAINER" bash -c "
cp $XSCHED_SRC/platforms/hip/shim/src/shim.cpp $XSCHED_SRC/platforms/hip/shim/src/shim.cpp.backup
echo '  ✅ 备份完成'
"

echo ""
echo "[2/5] 应用修复补丁..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC/platforms/hip/shim/src

# 修改 XLaunchKernel 函数
cat > /tmp/fix_shim.py << 'PYTHON'
import re

with open('shim.cpp', 'r') as f:
    content = f.read()

# 找到 XLaunchKernel 函数并修改
old_code = '''hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    XDEBG(\"XLaunchKernel: func=%p stream=%p\\\\n\", f, stream);
    if (stream == nullptr) {
        HipSyncBlockingXQueues();
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }'''

new_code = '''hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    XDEBG(\"XLaunchKernel: func=%p stream=%p\\\\n\", f, stream);
    
    // PATCH: 注释掉默认流的特殊处理，让所有流都走 XQueue 路径
    // if (stream == nullptr) {
    //     HipSyncBlockingXQueues();
    //     return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    // }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        // 如果没有 XQueue，使用同步方式调用原始 API
        HipSyncBlockingXQueues();
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }'''

content = content.replace(old_code, new_code)

with open('shim.cpp', 'w') as f:
    f.write(content)

print('  ✅ XLaunchKernel 已修改')
PYTHON

python3 /tmp/fix_shim.py
"

echo ""
echo "[3/5] 清理旧构建..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC
rm -rf build/*.so 2>/dev/null || true
echo '  ✅ 清理完成'
"

echo ""
echo "[4/5] 重新编译 XSched HIP 平台..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC
make hip 2>&1 | tail -50
" || {
    echo "  ❌ 编译失败"
    echo "  恢复备份..."
    docker exec "$CONTAINER" bash -c "
    cp $XSCHED_SRC/platforms/hip/shim/src/shim.cpp.backup $XSCHED_SRC/platforms/hip/shim/src/shim.cpp
    "
    exit 1
}

echo ""
echo "[5/5] 重新链接并复制库文件..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC/build/platforms/hip

# 重新编译 libhalhip.so（不使用版本脚本）
/usr/bin/c++ -fPIC -O3 -DRELEASE_MODE \
  -shared -Wl,-soname,libhalhip.so \
  -o libhalhip.so \
  CMakeFiles/halhip.dir/hal/src/hip_command.cpp.o \
  CMakeFiles/halhip.dir/hal/src/hip_queue.cpp.o \
  CMakeFiles/halhip.dir/hal/src/kernel_param.cpp.o \
  CMakeFiles/halhip.dir/hal/src/kernel_param_v2_8.cpp.o \
  CMakeFiles/halhip.dir/hal/src/kernel_param_v3.0.cpp.o \
  -Wl,-rpath,$XSCHED_BUILD/output/lib \
  ../../utils/libutils.a \
  ../../protocol/libprotocol.a \
  ../../preempt/libpreempt.so \
  -lpthread -ldl

# 重新链接 libshimhip.so
/usr/bin/c++ -fPIC -O3 -DRELEASE_MODE \
  -Wl,--exclude-libs,ALL \
  -Wl,--version-script=$XSCHED_SRC/platforms/hip/shim/hip_version.map \
  -shared -Wl,-soname,libshimhip.so \
  -o libshimhip.so \
  CMakeFiles/shimhip.dir/shim/src/intercept.cpp.o \
  CMakeFiles/shimhip.dir/shim/src/shim.cpp.o \
  -Wl,-rpath,$XSCHED_BUILD/output/lib \
  ../../utils/libutils.a \
  ../../protocol/libprotocol.a \
  libhalhip.so \
  ../../preempt/libpreempt.so \
  -lpthread -ldl

# 复制到 output 目录
mkdir -p $XSCHED_BUILD/output/lib
cp libhalhip.so $XSCHED_BUILD/output/lib/
cp libshimhip.so $XSCHED_BUILD/output/lib/
cp ../../preempt/libpreempt.so $XSCHED_BUILD/output/lib/

echo '  ✅ 库文件已复制'
ls -lh $XSCHED_BUILD/output/lib/*.so
"

echo ""
echo "========================================================================"
echo "✅ 修复完成！"
echo "========================================================================"
echo ""
echo "修改内容:"
echo "  - 注释掉了 XLaunchKernel 中默认流的特殊处理"
echo "  - 让所有流都走 XQueue 路径"
echo "  - 如果 XQueue 不存在，才调用 Driver::LaunchKernel"
echo ""
echo "下一步: 运行测试"
echo "  cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED"
echo "  ./debug_xsched.sh"
