
#include <vector>

__global__ void head2d_step(float* src, float* dest) {

};

int generateHeatMap(size_t xSize, size_t ySize, std::pair<size_t, size_t> heatSource, std::vector<float> &heatMap) {

    std::vector<float> resultHeat(heatMap.size(),0);

    float* dU, *dV; // device vectors

    float a = 0.5; // Diffusion Constant
    float dx = 0.1; // horizonal grid spacing
    float dy = 0.1; // vertical grid spacing
    const float dt = dx*dx * dy*dy / (2.0 * a * (dx*dx + dy*dy));


    // Copy nessary memory
    cudaMalloc((void **)&dU, heatMap.size() * sizeof(float));
    cudaMalloc((void **)&dV, resultHeat.size() * sizeof(float));
    cudaMemcpy(dU, heatMap.data(), heatMap.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, resultHeat.data(), resultHeat.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(xSize, ySize);
    dim3 blockSize(1, 1);

    head2d_step<<<gridSize, blockSize>>>(dU, dV);

    cudaDeviceSynchronize();
    cudaMemcpy(dV, resultHeat.data(), resultHeat.size() * sizeof(float), cudaMemcpyDeviceToHost);
    heatMap = resultHeat;

    cudaFree(dU);
    cudaFree(dV);

    return 1;
}