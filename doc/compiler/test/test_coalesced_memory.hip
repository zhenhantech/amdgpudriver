#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define N 4096
#define TILE 64

// 版本1: Naive转置（非合并写入）
__global__ void naive_transpose(float* out, float* in, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < n && y < n) {
        // 读取是合并的，但写入是非合并的！
        out[x * n + y] = in[y * n + x];
    }
}

// 版本2: 使用共享内存优化（合并访问）
__global__ void optimized_transpose(float* out, float* in, int n) {
    __shared__ float tile[TILE][TILE+1];  // +1 避免bank conflict
    
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    
    // 合并读取到共享内存
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    
    // 转置索引
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    // 合并写入
    if (x < n && y < n)
        out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

// 版本3: AoS数据结构（非合并）
struct ParticleAoS {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void update_aos(ParticleAoS* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].x += particles[i].vx * 0.01f;
        particles[i].y += particles[i].vy * 0.01f;
        particles[i].z += particles[i].vz * 0.01f;
    }
}

// 版本4: SoA数据结构（合并访问）
struct ParticlesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
};

__global__ void update_soa(ParticlesSoA particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles.x[i] += particles.vx[i] * 0.01f;
        particles.y[i] += particles.vy[i] * 0.01f;
        particles.z[i] += particles.vz[i] * 0.01f;
    }
}

double benchmark_kernel(void (*kernel)(float*, float*, int), 
                        float* d_out, float* d_in, int n, 
                        dim3 grid, dim3 block, int iterations = 100) {
    // 预热
    kernel<<<grid, block>>>(d_out, d_in, n);
    hipDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_out, d_in, n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Coalesced Memory Access 测试\n";
    std::cout << "========================================\n\n";
    
    // 测试1: 矩阵转置
    std::cout << "测试1: 矩阵转置 (" << N << "x" << N << ")\n";
    std::cout << "----------------------------------------\n";
    
    size_t matrix_bytes = N * N * sizeof(float);
    float *d_in, *d_out;
    hipMalloc(&d_in, matrix_bytes);
    hipMalloc(&d_out, matrix_bytes);
    
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (N + TILE-1)/TILE);
    
    // Naive版本
    double naive_ms = benchmark_kernel(naive_transpose, d_out, d_in, N, grid, block);
    double bandwidth_naive = (2.0 * matrix_bytes / 1e9) / (naive_ms / 1000);
    
    // Optimized版本
    double opt_ms = benchmark_kernel(optimized_transpose, d_out, d_in, N, grid, block);
    double bandwidth_opt = (2.0 * matrix_bytes / 1e9) / (opt_ms / 1000);
    
    std::cout << "Naive (非合并):     " << naive_ms << " ms, " 
              << bandwidth_naive << " GB/s\n";
    std::cout << "Optimized (合并):   " << opt_ms << " ms, " 
              << bandwidth_opt << " GB/s\n";
    std::cout << "加速比:             " << naive_ms / opt_ms << "x\n";
    std::cout << "带宽提升:           " << bandwidth_opt / bandwidth_naive << "x\n\n";
    
    hipFree(d_in);
    hipFree(d_out);
    
    // 测试2: AoS vs SoA
    std::cout << "测试2: 粒子更新 (AoS vs SoA)\n";
    std::cout << "----------------------------------------\n";
    
    const int num_particles = 1024 * 1024;
    
    // AoS版本
    ParticleAoS* d_particles_aos;
    hipMalloc(&d_particles_aos, num_particles * sizeof(ParticleAoS));
    
    dim3 block_particles(256);
    dim3 grid_particles((num_particles + 255) / 256);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        update_aos<<<grid_particles, block_particles>>>(d_particles_aos, num_particles);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double aos_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    // SoA版本
    ParticlesSoA h_particles_soa, d_particles_soa;
    hipMalloc(&d_particles_soa.x, num_particles * sizeof(float));
    hipMalloc(&d_particles_soa.y, num_particles * sizeof(float));
    hipMalloc(&d_particles_soa.z, num_particles * sizeof(float));
    hipMalloc(&d_particles_soa.vx, num_particles * sizeof(float));
    hipMalloc(&d_particles_soa.vy, num_particles * sizeof(float));
    hipMalloc(&d_particles_soa.vz, num_particles * sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        update_soa<<<grid_particles, block_particles>>>(d_particles_soa, num_particles);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double soa_ms = std::chrono::duration<double, std::milli>(end - start).count() / 100;
    
    std::cout << "AoS (非合并):       " << aos_ms << " ms\n";
    std::cout << "SoA (合并):         " << soa_ms << " ms\n";
    std::cout << "加速比:             " << aos_ms / soa_ms << "x\n\n";
    
    hipFree(d_particles_aos);
    hipFree(d_particles_soa.x);
    hipFree(d_particles_soa.y);
    hipFree(d_particles_soa.z);
    hipFree(d_particles_soa.vx);
    hipFree(d_particles_soa.vy);
    hipFree(d_particles_soa.vz);
    
    std::cout << "========================================\n";
    std::cout << "测试完成！\n";
    std::cout << "========================================\n";
    
    return 0;
}

