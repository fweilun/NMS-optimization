#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "nms.hpp"


const std::string BENCHMARK_DIR = "../benchmark/";


int const threadsPerBlock = sizeof(unsigned int) * 8; // 64 bits
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __constant__ unsigned long long BITMASK_TABLE[64] = {
    1ULL << 0,  1ULL << 1,  1ULL << 2,  1ULL << 3,
    1ULL << 4,  1ULL << 5,  1ULL << 6,  1ULL << 7,
    1ULL << 8,  1ULL << 9,  1ULL << 10, 1ULL << 11,
    1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15,
    1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19,
    1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23,
    1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27,
    1ULL << 28, 1ULL << 29, 1ULL << 30, 1ULL << 31,
    1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35,
    1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39,
    1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43,
    1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47,
    1ULL << 48, 1ULL << 49, 1ULL << 50, 1ULL << 51,
    1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55,
    1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59,
    1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63
};

template <typename T>
__device__ inline bool devIoU(
    T const* const a,
    T const* const b,
    const float threshold) {
  T left = max(a[0], b[0]), right = min(a[2], b[2]);
  T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
  float interS = (float)width * height;
  float Sa = ((float)a[2] - a[0]) * (a[3] - a[1]);
  float Sb = ((float)b[2] - b[0]) * (b[3] - b[1]);

  // (interS / (Sa + Sb - interS)) > threshold;
  // interS * (1+threshold) > threshold * Sa + threshold * Sb - threshold * interS;
  // return (interS / (Sa + Sb - interS)) > threshold;
  float lhs = interS * (1.0f + threshold);
  float rhs = threshold * (Sa + Sb);
  return lhs > rhs;
}

template <typename T>
__global__ void nms_kernel_impl(
    int n_boxes,
    double iou_threshold,
    const T* __restrict__ dev_boxes,
    unsigned int* __restrict__ dev_mask)          // 32‑bit mask
{
    const int row_tile = blockIdx.y;
    const int col_tile = blockIdx.x;
    if (row_tile > col_tile) return;               // upper‑triangular only

    const int row_sz = min(n_boxes - row_tile * threadsPerBlock, threadsPerBlock);
    const int col_sz = min(n_boxes - col_tile * threadsPerBlock, threadsPerBlock);

    // -------------------------------------------------
    // Load column tile into shared memory (once per block)
    // -------------------------------------------------
    __shared__ T tile_boxes[threadsPerBlock * 4];
    if (threadIdx.x < col_sz) {
        int gid = col_tile * threadsPerBlock + threadIdx.x;
        tile_boxes[threadIdx.x * 4 + 0] = dev_boxes[gid * 4 + 0];
        tile_boxes[threadIdx.x * 4 + 1] = dev_boxes[gid * 4 + 1];
        tile_boxes[threadIdx.x * 4 + 2] = dev_boxes[gid * 4 + 2];
        tile_boxes[threadIdx.x * 4 + 3] = dev_boxes[gid * 4 + 3];
    }
    __syncthreads();

    // -------------------------------------------------
    // Each thread processes one row box and builds a 32‑bit mask
    // -------------------------------------------------
    if (threadIdx.x < row_sz) {
        int row_gid = row_tile * threadsPerBlock + threadIdx.x;
        const T* cur_box = dev_boxes + row_gid * 4;
        unsigned int mask = 0U;
        int start = (row_tile == col_tile) ? threadIdx.x + 1 : 0;
        for (int j = start; j < col_sz; ++j) {
            if (devIoU<T>(cur_box, tile_boxes + j * 4,
                         static_cast<float>(iou_threshold))) {
                mask |= BITMASK_TABLE[j];          // j < 32 because threadsPerBlock==32
            }
        }
        const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
        dev_mask[row_gid * col_blocks + col_tile] = mask;
    }
}


__global__ static void gather_keep_from_mask(
    bool* __restrict__ keep,
    const unsigned int* __restrict__ dev_mask,
    const int n_boxes)
{
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);   // threadsPerBlock == 32
    const int tid       = threadIdx.x;                         // 0 … 31

    // 32‑bit per‑block removal status (one unsigned int per col‑block)
    extern __shared__ unsigned int removed[];
    for (int i = tid; i < col_blocks; i += blockDim.x) {
        removed[i] = 0U;
    }
    __syncthreads();

    for (int blk = 0; blk < col_blocks; ++blk) {
        // All threads see the same removed[blk] value
        __syncthreads();
        unsigned int removed_val = removed[blk];

        const int i_offset = blk * threadsPerBlock;
        if (i_offset >= n_boxes) break;

        #pragma unroll
        for (int lane = 0; lane < threadsPerBlock; ++lane) {
            const int idx = i_offset + lane;
            if (idx >= n_boxes) break;

            if (!(removed_val & BITMASK_TABLE[lane])) {   // not removed yet
                if (tid == 0) keep[idx] = true;           // one thread writes result

                const unsigned int* row_mask = dev_mask + idx * col_blocks;
                for (int j = tid; j < col_blocks; j += blockDim.x) {
                    removed[j] |= row_mask[j];
                }
                __syncthreads();                         // make the update visible
                removed_val = removed[blk];               // refresh for next lane
            }
        }
    }
}

std::vector<int> nms_kernel(const std::vector<Box>& boxes,
                            const std::vector<float>& scores,
                            double iou_threshold)
{
    const int n = static_cast<int>(boxes.size());
    if (n <= 0) return {};

    // 1) sort by score (descending) – keep original indices
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // 2) pack boxes in the sorted order (x1,y1,x2,y2)
    std::vector<float> h_boxes(4 * n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4*i + 0] = boxes[idx].x1;
        h_boxes[4*i + 1] = boxes[idx].y1;
        h_boxes[4*i + 2] = boxes[idx].x2;
        h_boxes[4*i + 3] = boxes[idx].y2;
    }

    // 3) device allocations (now unsigned int for mask)
    float* d_boxes = nullptr;
    unsigned int* d_mask = nullptr;
    bool* d_keep = nullptr;
    const int col_blocks = ceil_div(n, threadsPerBlock);
    cudaMalloc(&d_boxes, sizeof(float) * 4 * n);
    cudaMalloc(&d_mask,  sizeof(unsigned int) * n * col_blocks);
    cudaMalloc(&d_keep,  sizeof(bool) * n);

    cudaMemcpy(d_boxes, h_boxes.data(),
               sizeof(float) * 4 * n, cudaMemcpyHostToDevice);
    cudaMemset(d_mask, 0, sizeof(unsigned int) * n * col_blocks);
    cudaMemset(d_keep, 0, sizeof(bool) * n);

    // 4) launch kernels
    dim3 grid(col_blocks, col_blocks);
    
    int block_size = max(n / 32, 256);

    dim3 block(block_size);
    nms_kernel_impl<float><<<grid, 64>>>(
        n, iou_threshold, d_boxes, d_mask);
    cudaDeviceSynchronize();

    // shared‑memory size = col_blocks * sizeof(unsigned int)
    gather_keep_from_mask<<<
        1,
        block,
        col_blocks * sizeof(unsigned int)>>>(
        d_keep, d_mask, n);
    cudaDeviceSynchronize();

    // 5) copy back results
    std::vector<unsigned char> h_keep(n, 0);
    cudaMemcpy(h_keep.data(), d_keep,
               sizeof(bool) * n, cudaMemcpyDeviceToHost);

    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]) kept.push_back(order[i]);   // original index
    }

    // 6) cleanup
    cudaFree(d_boxes);
    cudaFree(d_mask);
    cudaFree(d_keep);
    return kept;
}