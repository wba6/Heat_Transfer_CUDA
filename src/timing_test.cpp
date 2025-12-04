#include "heat.cuh"
#include <chrono>
#include <cstdio>
#include <vector>

#define TIME_IT(name, code)                                                                    \
    do {                                                                                       \
        auto start = std::chrono::high_resolution_clock::now();                                \
        code;                                                                                  \
        auto end = std::chrono::high_resolution_clock::now();                                  \
        std::chrono::duration<double, std::milli> ms = end - start;                            \
        std::printf("% -20s: %f ms\n", name, ms.count());                                        \
    } while (0)

int main() {
    const int   nx    = 512;
    const int   ny    = 512;
    const float alpha = 1.0f;
    const float dx    = 1.0f;
    const float dt    = 0.24f * dx * dx / (4.0f * alpha);

    HeatSim sim;
    std::vector<float> h_init(nx * ny, 0.0f);
    std::vector<unsigned char> h_rgba(nx * ny * 4, 0);

    TIME_IT("heat_alloc", if (!heat_alloc(sim, nx, ny)) return 1);
    TIME_IT("heat_upload", heat_upload(sim, h_init.data()));
    TIME_IT("heat_step (1 iter)", heat_step(sim, alpha, dx, dt));
    TIME_IT("heat_step (10 iter)", for (int i = 0; i < 10; ++i) heat_step(sim, alpha, dx, dt));
    TIME_IT("heat_paint", heat_paint(sim, nx / 2, ny / 2, 20, 1.0f));
    TIME_IT("heat_to_rgba", heat_to_rgba(sim, h_rgba.data(), -1.0f, 1.0f));
    TIME_IT("heat_free", heat_free(sim));

    return 0;
}
