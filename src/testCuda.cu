#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "testCuda.hpp"

inline void checkCUDA(cudaError_t result, const char* expr, const char* file, int line)
{
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error at %s:%d for call '%s': %s\n",
                     file, line, expr, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(x) checkCUDA((x), #x, __FILE__, __LINE__)

// Simple vector add using shared memory
__global__ void vectorAddShared(const float* a,
                                const float* b,
                                float* c,
                                int n)
{
    extern __shared__ float shared[];
    float* as = shared;
    float* bs = shared + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        as[threadIdx.x] = a[idx];
        bs[threadIdx.x] = b[idx];
    }
    __syncthreads();

    if (idx < n) {
        c[idx] = as[threadIdx.x] + bs[threadIdx.x];
    }
}

// Simple kernel that scales unified memory
__global__ void scaleKernel(float* data, int n, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= alpha;
    }
}

// Helper to compare arrays
bool almostEqualArray(const float* a, const float* b, int n, float eps = 1e-5f)
{
    for (int i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            std::fprintf(stderr,
                         "Mismatch at index %d: %f vs %f\n",
                         i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int testCuda()
{
    std::printf("=== CUDA Sanity Check ===\n");

    // -------------------------------------------------------------------------
    // 1. Basic runtime / driver info
    // -------------------------------------------------------------------------
    int runtimeVersion = 0;
    int driverVersion  = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));

    std::printf("CUDA Runtime Version: %d\n", runtimeVersion);
    std::printf("CUDA Driver  Version: %d\n", driverVersion);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    std::printf("Found %d CUDA device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("  Device %d: %s\n", dev, prop.name);
        std::printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
        std::printf("    Total global mem  : %.2f GB\n",
                    static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("    MultiProcessorCount: %d\n", prop.multiProcessorCount);
    }

    // Use device 0 for tests
    CUDA_CHECK(cudaSetDevice(0));
    std::printf("\nUsing device 0 for tests.\n");

    // -------------------------------------------------------------------------
    // 2. Basic device memory alloc/copy test
    // -------------------------------------------------------------------------
    std::printf("\n[TEST] Device memory alloc & copy...\n");
    const int N = 1 << 12; // 4096
    const size_t bytes = N * sizeof(float);

    float* h_a = (float*)std::malloc(bytes);
    float* h_b = (float*)std::malloc(bytes);
    float* h_c = (float*)std::malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 3. Kernel + shared memory test
    // -------------------------------------------------------------------------
    std::printf("[TEST] Kernel launch + shared memory...\n");

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    size_t sharedBytes = 2 * blockSize * sizeof(float);

    vectorAddShared<<<gridSize, blockSize, sharedBytes>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Expected: h_c[i] = h_a[i] + h_b[i] = 3*i
    bool okVecAdd = true;
    for (int i = 0; i < N; ++i) {
        float expected = 3.0f * i;
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "Vector add mismatch at %d: got %f, expected %f\n",
                         i, h_c[i], expected);
            okVecAdd = false;
            break;
        }
    }

    std::printf("    Result: %s\n", okVecAdd ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // 4. Pinned (page-locked) host memory + async copy + streams
    // -------------------------------------------------------------------------
    std::printf("\n[TEST] Pinned memory + streams + async copy...\n");

    float* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes)); // page-locked host memory

    for (int i = 0; i < N; ++i) {
        h_pinned[i] = static_cast<float>(i);
    }

    float* d_streamData = nullptr;
    CUDA_CHECK(cudaMalloc(&d_streamData, bytes));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Async copy on stream1
    CUDA_CHECK(cudaMemcpyAsync(d_streamData, h_pinned, bytes,
                               cudaMemcpyHostToDevice, stream1));

    // Scale with a kernel on stream1
    scaleKernel<<<gridSize, blockSize, 0, stream1>>>(d_streamData, N, 2.0f);
    CUDA_CHECK(cudaGetLastError());

    // Copy back asynchronously on stream2
    CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_streamData, bytes,
                               cudaMemcpyDeviceToHost, stream2));

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    bool okPinned = true;
    for (int i = 0; i < N; ++i) {
        float expected = 2.0f * i;
        if (std::fabs(h_pinned[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "Pinned/stream mismatch at %d: got %f, expected %f\n",
                         i, h_pinned[i], expected);
            okPinned = false;
            break;
        }
    }

    std::printf("    Result: %s\n", okPinned ? "PASS" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_streamData));

    // -------------------------------------------------------------------------
    // 5. Unified memory test
    // -------------------------------------------------------------------------
    std::printf("\n[TEST] Unified memory (cudaMallocManaged)...\n");

    float* umem = nullptr;
    CUDA_CHECK(cudaMallocManaged(&umem, bytes));

    for (int i = 0; i < N; ++i) {
        umem[i] = 1.0f;
    }

    // Touch on device
    scaleKernel<<<gridSize, blockSize>>>(umem, N, 3.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool okUMem = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(umem[i] - 3.0f) > 1e-5f) {
            std::fprintf(stderr, "Unified memory mismatch at %d: got %f, expected 3.0\n",
                         i, umem[i]);
            okUMem = false;
            break;
        }
    }

    std::printf("    Result: %s\n", okUMem ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(umem));

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    std::printf("\n=== Summary ===\n");
    std::printf("Vector add + shared memory : %s\n", okVecAdd ? "PASS" : "FAIL");
    std::printf("Pinned + streams + async   : %s\n", okPinned ? "PASS" : "FAIL");
    std::printf("Unified memory             : %s\n", okUMem ? "PASS" : "FAIL");

    if (okVecAdd && okPinned && okUMem) {
        std::printf("\nAll tests PASSED. CUDA setup looks good.\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("\nSome tests FAILED. Check the messages above.\n");
        return EXIT_FAILURE;
    }
}
