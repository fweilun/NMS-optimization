#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>


const std::string BENCHMARK_DIR = "benchmark/";

struct Box { float x1, y1, x2, y2; };

std::vector<Box> read_boxes_csv(const std::string& path) {
    std::ifstream file(path);
    std::vector<Box> boxes;
    std::string line;
    // 跳過 header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int idx;
        Box b;
        char comma;
        ss >> idx >> comma >> b.x1 >> comma >> b.y1 >> comma >> b.x2 >> comma >> b.y2;
        boxes.push_back(b);
        // print_box(b);
    }
    return boxes;
}

std::vector<float> read_scores_csv(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> scores;
    std::string line;
    // 跳过 header
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int idx;
        char comma;
        float score;
        ss >> idx >> comma >> score;
        // printf("Score: %f\n", score);
        scores.push_back(score);
    }
    return scores;
}

std::vector<int> read_answer_csv(const std::string& path) {
    std::ifstream file(path);
    std::vector<int> out;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string idx, score;
        std::getline(ss, idx, ',');
        std::getline(ss, score, ',');
        out.push_back(std::stoi(score));
    }
    return out;
}

int const threadsPerBlock = sizeof(unsigned long long) * 8;
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
  return (interS / (Sa + Sb - interS)) > threshold;
}

template <typename T>
__global__ void nms_kernel_impl(
    int n_boxes,
    double iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const auto row_start = blockIdx.y;
  const auto col_start = blockIdx.x;

  if (row_start > col_start)
    return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncwarp();

  if (threadIdx.x < row_size) {
    const auto cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 4, iou_threshold)) {
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
  const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
  const auto thread_id = threadIdx.x;

  // Mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // Initialize removed.
  for (int i = thread_id; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; nblock++) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
    #pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; inblock++) {
      const int i = i_offset + inblock;
      if (i >= n_boxes)
        break;
      // Select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (thread_id == 0) {
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        // Remove all bboxes which overlap the candidate.
        for (int j = thread_id; j < col_blocks; j += blockDim.x) {
          removed[j] |= p[j];
        }
        __syncthreads();
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

    // 1) 按照 scores 排序（降序），就像 nms_kernel.cu 一样
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    // 2) 按照排序后的顺序重新排列 boxes
    std::vector<float> h_boxes(4 * n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4*i+0] = boxes[idx].x1;
        h_boxes[4*i+1] = boxes[idx].y1;
        h_boxes[4*i+2] = boxes[idx].x2;
        h_boxes[4*i+3] = boxes[idx].y2;
    }

    // 3) 配置 device buffer
    float* d_boxes = nullptr;
    const int col_blocks = ceil_div(n, threadsPerBlock);
    unsigned long long* d_mask = nullptr;
    bool* d_keep = nullptr;

    cudaMalloc(&d_boxes, sizeof(float) * 4 * n);
    cudaMalloc(&d_mask,  sizeof(unsigned long long) * n * col_blocks);
    cudaMalloc(&d_keep,  sizeof(bool) * n);

    cudaMemcpy(d_boxes, h_boxes.data(), sizeof(float) * 4 * n, cudaMemcpyHostToDevice);
    cudaMemset(d_mask, 0, sizeof(unsigned long long) * n * col_blocks);
    cudaMemset(d_keep, 0, sizeof(bool) * n);

    // 4) 啟動 kernels（使用預設 stream 0）
    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(threadsPerBlock);

    // 添加计时
    auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "=== NMS Kernel Overall Timing Started ===" << std::endl;

    // nms_kernel_impl 计时
    auto nms_impl_start = std::chrono::high_resolution_clock::now();
    // std::cout << "--- nms_kernel_impl timing started ---" << std::endl;
    
    nms_kernel_impl<float><<<blocks, threads>>>(
        n, iou_threshold, d_boxes, d_mask);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    auto nms_impl_end = std::chrono::high_resolution_clock::now();
    auto nms_impl_duration = std::chrono::duration_cast<std::chrono::microseconds>(nms_impl_end - nms_impl_start);
    std::cout << "--- nms_kernel_impl timing: " << nms_impl_duration.count() << " microseconds ---" << std::endl;

    // gather_keep_from_mask 计时
    auto gather_start = std::chrono::high_resolution_clock::now();
    // std::cout << "--- gather_keep_from_mask timing started ---" << std::endl;

    gather_keep_from_mask<<<
        1,
        std::min(col_blocks, threadsPerBlock),
        col_blocks * sizeof(unsigned long long)>>>(
        d_keep, d_mask, n);

    err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
      printf("success");
    }
    auto gather_end = std::chrono::high_resolution_clock::now();
    auto gather_duration = std::chrono::duration_cast<std::chrono::microseconds>(gather_end - gather_start);
    std::cout << "--- gather_keep_from_mask timing: " << gather_duration.count() << " microseconds ---" << std::endl;

    // 总体计时
    auto end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "=== NMS Kernel Overall Timing: " << overall_duration.count() << " microseconds ===" << std::endl;

    // 5) 取回 keep flags，組合保留的索引（按照原始顺序）
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

bool nms_test(int data_num, int id) {
    const std::string file_name = "rand_" + std::to_string(data_num) + "_" + std::to_string(id);
    const std::string data_path = BENCHMARK_DIR + file_name + ".csv";
    const std::string score_path = BENCHMARK_DIR + file_name + "_score.csv";
    const std::string answer_path = BENCHMARK_DIR + file_name + "_answer.csv";

    auto boxes_h = read_boxes_csv(data_path);
    auto scores_h = read_scores_csv(score_path);
    double iou_thr = 0.45;
    std::vector<int> result = nms_kernel(boxes_h, scores_h, iou_thr);
    std::vector<int> ans = read_answer_csv(answer_path);
    
    if (result.size() != ans.size()){
        printf("result size: %d  ans size: %d\n", result.size(), ans.size());
        return false;
    }
    int n = result.size();
    for (int i = 0;i < n; ++i) {
        if (result[i] != ans[i]) {
            printf("index %d is incorrect\n", i);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string n_str = argv[1];
    int n = stoi(n_str);
    if (nms_test(n, 0)) {
        printf("CORRECT!!\n");
    }
    return 0;
}