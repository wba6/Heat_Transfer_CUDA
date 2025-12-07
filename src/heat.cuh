#pragma once
#include <cstddef>

struct HeatSim {
    int nx{0}, ny{0};
    float* d_u{nullptr};
    float* d_v{nullptr};
    unsigned char* d_rgba{nullptr};
};

bool heat_alloc(HeatSim& sim, int nx, int ny);
void heat_free(HeatSim& sim);
void heat_upload(HeatSim& sim, const float* h_init);
void heat_step(HeatSim& sim, float alpha, float dx, float dt);
void heat_to_rgba(HeatSim& sim, unsigned char* h_rgba, float umin, float umax);
void heat_paint(HeatSim& sim, int cx, int cy, int radius, float amount);

// Kernel-level wrappers for timing
void heat_kernel_step_tiled(HeatSim& sim, float alpha, float dx, float dt);
void heat_kernel_copy_edges(HeatSim& sim);
void heat_kernel_to_rgba(HeatSim& sim, float umin, float umax);
void heat_kernel_paint(HeatSim& sim, int cx, int cy, int radius, float amount);
void heat_kernel_step_naive(HeatSim& sim, float alpha, float dx, float dt);


