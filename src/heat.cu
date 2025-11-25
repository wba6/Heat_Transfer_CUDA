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
    // Symmetric map: cold -> blue, neutral -> green, hot -> red.
    float span = fmaxf(fabsf(umin), fabsf(umax));
    span = fmaxf(span, 1e-6f); // avoid divide-by-zero
    float t = fminf(1.f, fmaxf(-1.f, u[k] / span)); // -1 cold, 0 neutral, 1 hot

    unsigned char r = 0, g = 0, b = 0;
    if (t < 0.0f) {
        // Blue to green
        float w = t + 1.0f; // 0 at coldest, 1 at neutral
        g = static_cast<unsigned char>(255.0f * w);
        b = static_cast<unsigned char>(255.0f * (1.0f - w));
    } else {
        // Green to red
        float w = t; // 0 at neutral, 1 at hottest
        r = static_cast<unsigned char>(255.0f * w);
        g = static_cast<unsigned char>(255.0f * (1.0f - w));
    }

    rgba[4*k+0] = r;
    rgba[4*k+1] = g;
    rgba[4*k+2] = b;
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
