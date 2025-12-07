#include "heat.cuh"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define TIME_HOST(name, code)                                                                  \
    do {                                                                                       \
        auto start = std::chrono::high_resolution_clock::now();                                \
        code;                                                                                  \
        auto end = std::chrono::high_resolution_clock::now();                                  \
        std::chrono::duration<double, std::milli> ms = end - start;                            \
        std::printf("%-30s: %f ms\n", name, ms.count());                                        \
    } while (0)

#define TIME_KERNEL(name, code)                                                                \
    do {                                                                                       \
        cudaEvent_t start, stop;                                                               \
        cudaEventCreate(&start);                                                               \
        cudaEventCreate(&stop);                                                                \
        cudaEventRecord(start);                                                                \
        code;                                                                                  \
        cudaEventRecord(stop);                                                                 \
        cudaEventSynchronize(stop);                                                            \
        float ms = 0;                                                                          \
        cudaEventElapsedTime(&ms, start, stop);\
        std::printf("%-30s: %f ms\n", name, ms);                                                \
        cudaEventDestroy(start);                                                               \
        cudaEventDestroy(stop);                                                                \
    } while (0)

int main() {
    const int   nx    = 512;
    const int   ny    = 512;
    const float alpha = 1.0f;
    const float dx    = 1.0f;
    const float dt    = 0.24f * dx * dx / (4.0f * alpha);
    const size_t n_bytes = static_cast<size_t>(nx) * ny * sizeof(float);
    const size_t rgba_bytes = n_bytes * 4;

    HeatSim sim;
    std::vector<float> h_init(nx * ny, 0.0f);
    std::vector<unsigned char> h_rgba(nx * ny * 4, 0);

    std::printf("--- Host Function Timings ---\n");
    TIME_HOST("heat_alloc", if (!heat_alloc(sim, nx, ny)) return 1);
    TIME_HOST("heat_upload", heat_upload(sim, h_init.data()));
    TIME_HOST("heat_step (1 iter)", heat_step(sim, alpha, dx, dt));
    TIME_HOST("heat_step (10 iter)", for (int i = 0; i < 10; ++i) heat_step(sim, alpha, dx, dt));
    TIME_HOST("heat_paint", heat_paint(sim, nx / 2, ny / 2, 20, 1.0f));
    TIME_HOST("heat_to_rgba", heat_to_rgba(sim, h_rgba.data(), -1.0f, 1.0f));
    
    // Re-initialize for kernel timings
    heat_free(sim);
    heat_alloc(sim, nx, ny);
    heat_upload(sim, h_init.data());

    std::printf("\n--- Individual Kernel Timings ---\n");
    TIME_KERNEL("k_heat_step_tiled", heat_kernel_step_tiled(sim, alpha, dx, dt));
    TIME_KERNEL("k_heat_step_naive", heat_kernel_step_naive(sim, alpha, dx, dt));
    TIME_KERNEL("k_copy_edges", heat_kernel_copy_edges(sim));
    TIME_KERNEL("k_to_rgba", heat_kernel_to_rgba(sim, -1.0f, 1.0f));
    TIME_KERNEL("k_paint", heat_kernel_paint(sim, nx / 2, ny / 2, 20, 1.0f));

    std::printf("\n--- Memory Copy Timings ---\n");
    TIME_KERNEL("H->D (upload)", cudaMemcpy(sim.d_u, h_init.data(), n_bytes, cudaMemcpyHostToDevice));
    TIME_KERNEL("D->H (download rgba)", cudaMemcpy(sim.d_rgba, h_rgba.data(), rgba_bytes, cudaMemcpyDeviceToHost));


    TIME_HOST("heat_free", heat_free(sim));

    return 0;
}
