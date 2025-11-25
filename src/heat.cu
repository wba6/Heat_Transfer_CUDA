#include <cuda_runtime.h>
#include "heat.cuh"

// 5-point Laplacian explicit step
__global__ void k_heat_step(const float* __restrict__ u, float* __restrict__ v,
                            int nx, int ny, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<=0 || j<=0 || i>=nx-1 || j>=ny-1) return;
    int idx = j*nx + i;
    float uc = u[idx];
    float lap = u[idx-1] + u[idx+1] + u[idx-nx] + u[idx+nx] - 4.0f*uc;
    v[idx] = uc + r * lap;
}

__global__ void k_copy_edges(float* v, const float* u, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<nx && j<ny) {
        if (i==0 || j==0 || i==nx-1 || j==ny-1) {
            v[j*nx + i] = 0.0f; // Dirichlet boundary: fixed 0
        }
    }
}

__global__ void k_to_rgba(const float* u, unsigned char* rgba, int n, float umin, float umax) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k>=n) return;
    // Hue-based map: cold -> blue, hot -> red, full brightness (no black gaps).
    float span = umax - umin;
    span = fmaxf(span, 1e-6f);
    float t = (u[k] - umin) / span;        // 0 cold, 1 hot
    t = fminf(1.f, fmaxf(0.f, t));

    // HSV to RGB with S=1, V=1, hue from 240° (blue) to 0° (red)
    float h = (1.0f - t) * (2.0f / 3.0f);  // 0..1
    float hf = h * 6.0f;
    int   i  = (int)hf;
    float f  = hf - i;
    float p = 0.0f;
    float q = 1.0f - f;
    float r=0,g=0,b=0;
    switch (i % 6) {
        case 0: r = 1.0f; g = f;     b = 0.0f; break;
        case 1: r = q;    g = 1.0f;  b = 0.0f; break;
        case 2: r = 0.0f; g = 1.0f;  b = f;    break;
        case 3: r = 0.0f; g = q;     b = 1.0f; break;
        case 4: r = f;    g = 0.0f;  b = 1.0f; break;
        case 5: r = 1.0f; g = 0.0f;  b = q;    break;
    }

    rgba[4*k+0] = static_cast<unsigned char>(255.0f * r);
    rgba[4*k+1] = static_cast<unsigned char>(255.0f * g);
    rgba[4*k+2] = static_cast<unsigned char>(255.0f * b);
    rgba[4*k+3] = 255;
}

__global__ void k_paint(float* u, int nx, int ny, int cx, int cy, int r, float amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=nx || j>=ny) return;
    int dx = i - cx;
    int dy = j - cy;
    if (dx*dx + dy*dy <= r*r) {
        float current = u[j*nx + i];
        // Paint pulls values toward the requested amount so cold stays blue and hot stays red.
        if (amount >= 0.0f) {
            u[j*nx + i] = fmaxf(current, amount);
        } else {
            u[j*nx + i] = fminf(current, amount);
        }
    }
}

bool heat_alloc(HeatSim& sim, int nx, int ny) {
    sim.nx = nx; sim.ny = ny;
    size_t n = static_cast<size_t>(nx) * ny;
    if (cudaMalloc(&sim.d_u, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&sim.d_v, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&sim.d_rgba, n * 4 * sizeof(unsigned char)) != cudaSuccess) return false;
    cudaMemset(sim.d_u, 0, n*sizeof(float));
    cudaMemset(sim.d_v, 0, n*sizeof(float));
    return true;
}

void heat_free(HeatSim& sim) {
    cudaFree(sim.d_rgba);
    cudaFree(sim.d_v);
    cudaFree(sim.d_u);
    sim = HeatSim{};
}

void heat_upload(HeatSim& sim, const float* h_init) {
    size_t n = static_cast<size_t>(sim.nx) * sim.ny;
    cudaMemcpy(sim.d_u, h_init, n*sizeof(float), cudaMemcpyHostToDevice);
}

void heat_step(HeatSim& sim, float alpha, float dx, float dt) {
    float r = alpha * dt / (dx*dx); // assume dx == dy
    dim3 block(16,16);
    dim3 grid((sim.nx + block.x - 1)/block.x, (sim.ny + block.y - 1)/block.y);
    k_heat_step<<<grid, block>>>(sim.d_u, sim.d_v, sim.nx, sim.ny, r);
    k_copy_edges<<<grid, block>>>(sim.d_v, sim.d_u, sim.nx, sim.ny);
    // swap
    float* tmp = sim.d_u; sim.d_u = sim.d_v; sim.d_v = tmp;
}

void heat_to_rgba(HeatSim& sim, unsigned char* h_rgba, float umin, float umax) {
    size_t n = static_cast<size_t>(sim.nx) * sim.ny;
    int threads = 256;
    int blocks = (int)((n + threads - 1)/threads);
    k_to_rgba<<<blocks, threads>>>(sim.d_u, sim.d_rgba, (int)n, umin, umax);
    cudaMemcpy(h_rgba, sim.d_rgba, n*4*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void heat_paint(HeatSim& sim, int cx, int cy, int radius, float amount) {
    dim3 block(16,16);
    dim3 grid((sim.nx + block.x - 1)/block.x, (sim.ny + block.y - 1)/block.y);
    k_paint<<<grid, block>>>(sim.d_u, sim.nx, sim.ny, cx, cy, radius, amount);
}
