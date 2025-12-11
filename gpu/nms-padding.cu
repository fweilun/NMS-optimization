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


int const threadsPerBlock = sizeof(unsigned long long) * 8; // 64 bits
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ inline unsigned long long devIoU(
    const float* a,
    const float* b,
    float threshold) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left, 0.0), height = max(bottom - top, 0.0);
  float interS = width * height;
  float Sa = (a[2] - a[0]) * (a[3] - a[1]);
  float Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return (interS / (Sa + Sb - interS)) > threshold;
}


__global__ void nms_kernel_impl(
    int n_boxes,
    double iou_threshold,
    const float* dev_boxes,
    unsigned long long* dev_mask) {
  
  const auto row_start = blockIdx.y;
  const auto col_start = blockIdx.x;
  if (row_start > col_start)
    return;

  const int row_size = threadsPerBlock;
  const int col_size = threadsPerBlock;

  __shared__ float block_boxes[threadsPerBlock * 4];
  block_boxes[threadIdx.x * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
  block_boxes[threadIdx.x * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
  block_boxes[threadIdx.x * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
  block_boxes[threadIdx.x * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  __syncthreads();

  if (threadIdx.x < row_size) {
    const auto cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      t |= (devIoU(cur_box, block_boxes + i * 4, iou_threshold) << i);
      if (devIoU(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

__global__ static void gather_keep_from_mask(
    bool* keep,
    const unsigned long long* dev_mask,
    const int n_boxes) {
  // Taken and adapted from mmcv
  // https://github.com/open-mmlab/mmcv/blob/03ce9208d18c0a63d7ffa087ea1c2f5661f2441a/mmcv/ops/csrc/common/cuda/nms_cuda_kernel.cuh#L76
  const int col_blocks = n_boxes/ threadsPerBlock;
  const auto thread_id = threadIdx.x;

  extern __shared__ unsigned long long removed[];

  // Initialize removed.
  for (int i = thread_id; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncwarp();
  for (int nblock = 0; nblock < col_blocks; nblock++) {
    auto removed_val = removed[nblock];
    __syncwarp();
    const int i_offset = nblock * threadsPerBlock;
    #pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; inblock++) {
      const int i = i_offset + inblock;
      if (!(removed_val & (1ULL << inblock))) {
        if (thread_id == 0) {
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        for (int j = thread_id; j < col_blocks; j += blockDim.x) {
          removed[j] |= p[j];
        }
        __syncwarp();
        removed_val = removed[nblock];
      }
    }
  }
}

void save_kept_to_csv(std::vector<int>& kept, const std::string& filename)
{
    std::sort(kept.begin(), kept.end());
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << " for writing.\n";
        return;
    }

    // 寫表頭（可選）
    file << "index\n";

    // 寫內容
    for (int idx : kept)
        file << idx << "\n";

    file.close();
    std::cout << "✅ Saved " << kept.size() << " indices to " << filename << std::endl;
}

std::vector<int> nms_kernel(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold)
{
    const int n = (int)boxes.size();
    if (n <= 0) return {};

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0); // initialize with 0, 1, ..., n-1
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    // 2) 按照排序后的顺序重新排列 boxes

    int n_padding = (n + 63) / 64 * 64;
    std::vector<float> h_boxes(4 * n_padding, 0.0);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4*i+0] = boxes[idx].x1;
        h_boxes[4*i+1] = boxes[idx].y1;
        h_boxes[4*i+2] = boxes[idx].x2;
        h_boxes[4*i+3] = boxes[idx].y2;
    }

    float* d_boxes = nullptr;
    const int col_blocks = ceil_div(n, threadsPerBlock); // threadsPerBlock = 64
    unsigned long long* d_mask = nullptr;
    bool* d_keep = nullptr;

    cudaMalloc(&d_boxes, sizeof(float) * 4 * n_padding);
    cudaMalloc(&d_mask,  sizeof(unsigned long long) * n_padding * col_blocks);
    cudaMalloc(&d_keep,  sizeof(bool) * n);

    cudaMemcpy(d_boxes, h_boxes.data(), sizeof(float) * 4 * n_padding, cudaMemcpyHostToDevice);
    cudaMemset(d_mask, 0, sizeof(unsigned long long) * n_padding * col_blocks);
    cudaMemset(d_keep, 0, sizeof(bool) * n_padding);

    // 4) 啟動 kernels（使用預設 stream 0）
    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(threadsPerBlock);
    
    nms_kernel_impl<<<blocks, threads>>>(
        n_padding, iou_threshold, d_boxes, d_mask);
    
    cudaError_t err = cudaDeviceSynchronize();

    gather_keep_from_mask<<<
        1,                                            // 1 block
        std::min(col_blocks, threadsPerBlock),        // up to 'threadsPerBlock' = 64 threads
        col_blocks * sizeof(unsigned long long)>>>(   // shared memory size
        d_keep, d_mask, n_padding);
    err = cudaDeviceSynchronize();

    std::vector<unsigned char> h_keep(n, 0);
    cudaMemcpy(h_keep.data(), d_keep, sizeof(bool) * n, cudaMemcpyDeviceToHost);

    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]) {
            kept.push_back(order[i]);  // 返回原始索引
        }
    }
    save_kept_to_csv(kept, "tempsave_standard.csv");

    // 6) 清理
    cudaFree(d_boxes);
    cudaFree(d_mask);
    cudaFree(d_keep);
    return kept;
}
