#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <chrono>
#include <thread>

// 简单的测试kernel
__global__ void dummy_kernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = idx * 2;
    }
}

class StreamTester {
private:
    int num_streams_;
    std::vector<hipStream_t> streams_;
    std::vector<int*> d_buffers_;
    const int buffer_size_ = 1024;
    
public:
    StreamTester(int num_streams) : num_streams_(num_streams) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Stream Tester Initialization" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Number of Streams: " << num_streams_ << std::endl;
        std::cout << "Process ID: " << getpid() << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    ~StreamTester() {
        cleanup();
    }
    
    bool createStreams() {
        std::cout << "[CREATE] Creating " << num_streams_ << " streams..." << std::endl;
        
        streams_.resize(num_streams_);
        d_buffers_.resize(num_streams_);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_streams_; i++) {
            hipError_t err = hipStreamCreate(&streams_[i]);
            if (err != hipSuccess) {
                std::cerr << "[ERROR] Failed to create stream " << i 
                         << ": " << hipGetErrorString(err) << std::endl;
                return false;
            }
            
            // 为每个stream分配device memory
            err = hipMalloc(&d_buffers_[i], buffer_size_ * sizeof(int));
            if (err != hipSuccess) {
                std::cerr << "[ERROR] Failed to allocate memory for stream " << i 
                         << ": " << hipGetErrorString(err) << std::endl;
                return false;
            }
            
            if ((i + 1) % 8 == 0 || i == num_streams_ - 1) {
                std::cout << "[CREATE] Created " << (i + 1) << "/" << num_streams_ 
                         << " streams" << std::endl;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "[CREATE] ✓ Successfully created " << num_streams_ 
                 << " streams in " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    
    void launchKernels() {
        std::cout << "\n[LAUNCH] Launching kernels on all streams..." << std::endl;
        
        dim3 block(256);
        dim3 grid((buffer_size_ + block.x - 1) / block.x);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_streams_; i++) {
            hipLaunchKernelGGL(dummy_kernel, grid, block, 0, streams_[i],
                              d_buffers_[i], buffer_size_);
        }
        
        // 同步所有streams
        for (int i = 0; i < num_streams_; i++) {
            hipStreamSynchronize(streams_[i]);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "[LAUNCH] ✓ All kernels completed in " << duration.count() 
                 << " ms" << std::endl;
    }
    
    void testConcurrentSubmission() {
        std::cout << "\n[CONCURRENT] Testing concurrent kernel submission..." << std::endl;
        
        dim3 block(256);
        dim3 grid((buffer_size_ + block.x - 1) / block.x);
        
        // 连续提交kernel到所有streams（不等待）
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int round = 0; round < 5; round++) {
            for (int i = 0; i < num_streams_; i++) {
                hipLaunchKernelGGL(dummy_kernel, grid, block, 0, streams_[i],
                                  d_buffers_[i], buffer_size_);
            }
        }
        
        // 一次性同步所有streams
        for (int i = 0; i < num_streams_; i++) {
            hipStreamSynchronize(streams_[i]);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "[CONCURRENT] ✓ Submitted " << (num_streams_ * 5) 
                 << " kernels (" << num_streams_ << " streams × 5 rounds) in " 
                 << duration.count() << " ms" << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Process ID:        " << getpid() << std::endl;
        std::cout << "Streams Created:   " << num_streams_ << std::endl;
        std::cout << "Expected AQL Queues: " << num_streams_ << " (1 per stream)" << std::endl;
        std::cout << "Expected HQD Usage:  " << std::min(num_streams_, 32) << "/32" << std::endl;
        
        if (num_streams_ <= 32) {
            std::cout << "HQD Status:        ✓ Sufficient (each queue gets dedicated HQD)" << std::endl;
        } else {
            std::cout << "HQD Status:        ⚠ Insufficient (HQD sharing required)" << std::endl;
            std::cout << "Sharing Ratio:     " << (float)num_streams_ / 32.0 << ":1" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
    
private:
    void cleanup() {
        std::cout << "\n[CLEANUP] Destroying streams and freeing memory..." << std::endl;
        
        for (size_t i = 0; i < streams_.size(); i++) {
            if (d_buffers_[i]) {
                hipFree(d_buffers_[i]);
            }
            if (streams_[i]) {
                hipStreamDestroy(streams_[i]);
            }
        }
        
        std::cout << "[CLEANUP] ✓ Cleanup completed" << std::endl;
    }
};

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <num_streams> [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  <num_streams>    Number of streams to create (e.g., 16, 32, 64)" << std::endl;
    std::cout << "  -w, --wait       Wait after creation for inspection (default: 5s)" << std::endl;
    std::cout << "  -t, --time <sec> Wait time in seconds" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << prog_name << " 16        # Test with 16 streams" << std::endl;
    std::cout << "  " << prog_name << " 32 -w     # Test with 32 streams and wait" << std::endl;
    std::cout << "  " << prog_name << " 64 -t 10  # Test with 64 streams and wait 10s" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    int num_streams = std::atoi(argv[1]);
    if (num_streams <= 0 || num_streams > 1024) {
        std::cerr << "Error: Invalid number of streams (must be 1-1024)" << std::endl;
        return 1;
    }
    
    bool wait_after_create = false;
    int wait_time = 5;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--wait") {
            wait_after_create = true;
        } else if (arg == "-t" || arg == "--time") {
            if (i + 1 < argc) {
                wait_time = std::atoi(argv[++i]);
                wait_after_create = true;
            }
        }
    }
    
    // 设置GPU设备
    hipSetDevice(0);
    
    // 打印设备信息
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU Device Information" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Device Name: " << props.name << std::endl;
    std::cout << "Compute Units: " << props.multiProcessorCount << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 创建测试器
    StreamTester tester(num_streams);
    
    // 创建streams
    if (!tester.createStreams()) {
        std::cerr << "Failed to create streams" << std::endl;
        return 1;
    }
    
    // 如果需要等待，暂停以便检查dmesg
    if (wait_after_create) {
        std::cout << "\n[WAIT] Waiting " << wait_time << " seconds for inspection..." << std::endl;
        std::cout << "[WAIT] You can now check dmesg in another terminal:" << std::endl;
        std::cout << "[WAIT]   sudo dmesg | grep 'CREATE_QUEUE' | tail -" << num_streams << std::endl;
        std::cout << "[WAIT]   sudo dmesg | grep 'hqd slot' | tail -" << num_streams << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(wait_time));
    }
    
    // 测试kernel提交
    tester.launchKernels();
    
    // 测试并发提交
    tester.testConcurrentSubmission();
    
    // 打印总结
    tester.printSummary();
    
    std::cout << "\n[INFO] Test completed successfully!" << std::endl;
    std::cout << "[INFO] Check dmesg for detailed queue creation logs:" << std::endl;
    std::cout << "[INFO]   sudo dmesg | grep -E 'CREATE_QUEUE|hqd slot' | tail -50" << std::endl;
    
    return 0;
}
