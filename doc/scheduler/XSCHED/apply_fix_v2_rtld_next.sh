#!/bin/bash
# XSched 修复 v2：使用 RTLD_NEXT 避免符号查找循环
set -e

echo "========================================================================"
echo "XSched 修复 v2：使用 RTLD_NEXT"
echo "========================================================================"
echo ""

XSCHED_SRC="/data/dockercode/xsched-official"
XSCHED_BUILD="/data/dockercode/xsched-build"
CONTAINER="zhenflashinfer_v1"

echo "[1/5] 恢复原始文件并创建新修改..."
docker exec "$CONTAINER" bash -c "
cp $XSCHED_SRC/platforms/hip/shim/src/shim.cpp.backup $XSCHED_SRC/platforms/hip/shim/src/shim.cpp
echo '  ✅ 恢复原始文件'
"

echo ""
echo "[2/5] 在 shim.cpp 中添加 RTLD_NEXT fallback..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC/platforms/hip/shim/src

cat > /tmp/add_rtld_next.py << 'PYTHON'
content = open('shim.cpp').read()

# 在文件开头添加 fallback 函数
header_code = '''#include \"xsched/hip/shim/shim.h\"
#include \"xsched/hip/hal/hip_command.h\"
#include \"xsched/hip/hal/hip_queue.h\"
#include \"xsched/hip/hal/handle.h\"
#include <memory>

using namespace xsched;
using namespace xsched::hip;
using namespace xsched::preempt;

// PATCH: 添加直接调用原始 HIP API 的函数（使用 RTLD_NEXT）
static hipError_t ORIGINAL_hipLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, 
                                            void **args, size_t sharedMemBytes, hipStream_t stream)
{
    typedef hipError_t (*LaunchKernelFunc)(const void*, dim3, dim3, void**, size_t, hipStream_t);
    static LaunchKernelFunc original_func = nullptr;
    
    if (original_func == nullptr) {
        // 使用 RTLD_NEXT 查找下一个库中的 hipLaunchKernel
        original_func = (LaunchKernelFunc)dlsym(RTLD_NEXT, \"hipLaunchKernel\");
        if (original_func == nullptr) {
            XERRO(\"Failed to find original hipLaunchKernel: %s\", dlerror());
            return hipErrorNotFound;
        }
        XINFO(\"Found original hipLaunchKernel at %p\", original_func);
    }
    
    return original_func(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

'''

# 替换原来的 include 部分
old_includes = '''#include \"xsched/hip/shim/shim.h\"
#include \"xsched/hip/hal/hip_command.h\"
#include \"xsched/hip/hal/hip_queue.h\"
#include \"xsched/hip/hal/handle.h\"
#include <memory>

using namespace xsched;
using namespace xsched::hip;
using namespace xsched::preempt;'''

content = content.replace(old_includes, header_code)

# 修改 XLaunchKernel 使用 ORIGINAL_hipLaunchKernel
old_fallback = 'return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);'
new_fallback = 'return ORIGINAL_hipLaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);'

content = content.replace(old_fallback, new_fallback)

open('shim.cpp', 'w').write(content)
print('  ✅ 已添加 RTLD_NEXT fallback')
PYTHON

python3 /tmp/add_rtld_next.py
"

echo ""
echo "[3/5] 清理并重新编译..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC
rm -rf build/*.so 2>/dev/null || true
make hip 2>&1 | tail -50
"

if [ $? -ne 0 ]; then
    echo "  ❌ 编译失败"
    exit 1
fi

echo ""
echo "[4/5] 重新链接库文件..."
docker exec "$CONTAINER" bash -c "
cd $XSCHED_SRC/build/platforms/hip

# libhalhip.so (不使用版本脚本)
/usr/bin/c++ -fPIC -O3 -DRELEASE_MODE \
  -shared -Wl,-soname,libhalhip.so \
  -o libhalhip.so \
  CMakeFiles/halhip.dir/hal/src/*.o \
  -Wl,-rpath,$XSCHED_BUILD/output/lib \
  ../../utils/libutils.a ../../protocol/libprotocol.a ../../preempt/libpreempt.so \
  -lpthread -ldl

# libshimhip.so (链接 libhalhip.so)
/usr/bin/c++ -fPIC -O3 -DRELEASE_MODE \
  -Wl,--exclude-libs,ALL \
  -Wl,--version-script=$XSCHED_SRC/platforms/hip/shim/hip_version.map \
  -shared -Wl,-soname,libshimhip.so \
  -o libshimhip.so \
  CMakeFiles/shimhip.dir/shim/src/*.o \
  -Wl,-rpath,$XSCHED_BUILD/output/lib \
  ../../utils/libutils.a ../../protocol/libprotocol.a libhalhip.so ../../preempt/libpreempt.so \
  -lpthread -ldl

mkdir -p $XSCHED_BUILD/output/lib
cp libhalhip.so libshimhip.so ../../preempt/libpreempt.so $XSCHED_BUILD/output/lib/
echo '  ✅ 库文件已复制'
"

echo ""
echo "[5/5] 测试修复..."
docker exec "$CONTAINER" bash -c "
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
python3 << 'EOF'
import torch
print('Testing XSched with RTLD_NEXT fix...')
print('Step 1: Basic tensor on GPU')
try:
    a = torch.randn(10, 10, device='cuda:0')
    print(f'  ✅ SUCCESS: {a.shape}')
    
    b = torch.randn(10, 10, device='cuda:0')
    c = a + b
    torch.cuda.synchronize()
    print(f'  ✅ Addition works')
    
    print('\\n🎉🎉🎉 XSched FIX SUCCESSFUL!')
except Exception as e:
    print(f'  ❌ Still fails: {e}')
    import sys
    sys.exit(1)
EOF
" 2>&1 | tail -30

echo ""
echo "========================================================================"
echo "修复脚本执行完毕"
echo "========================================================================"
