#!/bin/bash
# partition_info.sh - 查看 GPU 分区配置
# 使用方法: bash partition_info.sh

echo "============================================"
echo "GPU Partition Configuration Info"
echo "============================================"
echo

for card in /sys/class/drm/card*/device/current_compute_partition; do
    card_dir=$(dirname $card)
    card_name=$(basename $(dirname $card_dir))
    
    echo "=== $card_name ==="
    echo "Compute Partition: $(cat $card_dir/current_compute_partition)"
    echo "Available Compute: $(cat $card_dir/available_compute_partition)"
    echo "Memory Partition:  $(cat $card_dir/current_memory_partition)"
    echo "Available Memory:  $(cat $card_dir/available_memory_partition)"
    
    # 需要 root 权限的高级信息
    if [ -r $card_dir/compute_partition_config/xcc/num_inst ]; then
        echo "XCC Instances:     $(cat $card_dir/compute_partition_config/xcc/num_inst)"
        echo "SDMA Engines:      $(cat $card_dir/compute_partition_config/dma/num_inst)"
        echo "DEC Engines:       $(cat $card_dir/compute_partition_config/dec/num_inst)"
        echo "JPEG Decoders:     $(cat $card_dir/compute_partition_config/jpeg/num_inst)"
    fi
    echo
done

# 查看 render 节点分布
echo "=== Render Nodes Distribution ==="
for i in {0..7}; do
    start=$((128 + i*8))
    end=$((start + 7))
    if [ -e /dev/dri/renderD$start ]; then
        echo "GPU $i: renderD$start - renderD$end"
    fi
done

echo
echo "============================================"
echo "Total Render Nodes: $(ls /dev/dri/renderD* 2>/dev/null | wc -l)"
echo "============================================"

