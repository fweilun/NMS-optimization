#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>


#define TILE_SIZE 64
#define threadsPerWarp 32
#define MaxLoadPerThread 16

const std::string BENCHMARK_DIR = "benchmark/";

// Helper function for ceiling division
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct Box {
    float x1, y1, x2, y2;
};

// void print_box(Box b) {
//     printf("box: %f %f %f %f\n", b.x1, b.y1 ,b.x2 ,b.y2);
// }

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
    while (std::getline(file, line)) 
        std::stringstream ss(line);
        int idx;
        char comma;
        float score;
        ss >> idx >> comma >> score;
        scores.push_back(score);
    
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
    // printf("%f\n", (interS / (Sa + Sb - interS)));
    return (interS / (Sa + Sb - interS)) > threshold;
}

__global__ void nms_kernel_1d(
    const int n,
    const float* boxes,
    const float iou_thresh,
    bool* keep)
{
    // 32 個thread處理的資料總數: 2048 (TILE_SIZE * threadsPerWarp)
    __shared__ float row_boxes[TILE_SIZE * threadsPerWarp * 4]; // 4: box
    // __shared__ unsigned long long removed[threadsPerWarp];

    int tx = threadIdx.x;
    
    for (int i = 0;i < MaxLoadPerThread; ++i) {
        int temp_idx = i * blockDim.x + tx;
        // int temp_idx = tx * MaxLoadPerThread + i;
        if (temp_idx >= n) break;
        row_boxes[temp_idx * 4] = boxes[temp_idx * 4];
        row_boxes[temp_idx * 4 + 1] = boxes[temp_idx * 4 + 1];
        row_boxes[temp_idx * 4 + 2] = boxes[temp_idx * 4 + 2];
        row_boxes[temp_idx * 4 + 3] = boxes[temp_idx * 4 + 3];
    }
    __syncthreads();

    if (tx < 32) {
        int col_start = tx * TILE_SIZE;
        
        unsigned mask = 0xffffffff;
        
        // 每個thread都處理64個index，代表這些index要不要丟棄由他決定。
        // 一開始都設為0，代表所有index不用丟棄。
        // 他會一直保留local_removed 直到iter到他這一輪。
        unsigned long long local_removed = 0ULL;
        unsigned long long row_removed = 0ULL;

        for (int iter = 0; iter < n; ++iter) {
            int hot_thread = iter / TILE_SIZE;
            
            // 64 bits warp communication is not supported for now.
            unsigned int low  = static_cast<unsigned int>(local_removed & 0xFFFFFFFFu);
            unsigned int high = static_cast<unsigned int>(local_removed >> 32);
            low  = __shfl_sync(mask, low,  hot_thread);
            high = __shfl_sync(mask, high, hot_thread);
            row_removed = (static_cast<unsigned long long>(high) << 32) | low;
            // row_removed = removed[hot_thread];
            // 64 bits

            // 該 iter 已經被刪除
            if (row_removed & (1ULL << (iter % TILE_SIZE))) continue;
            
            // 2. 接下來是 iter 不能跳過，就往右看64個bits
            #pragma unroll
            for (int in_block = 0; in_block < TILE_SIZE; ++in_block) {
                int i = col_start + in_block;
                if (i==iter) {
                    keep[i] = true;
                }
                // in_block bit has been removed.
                unsigned long long iou_result = devIoU(&row_boxes[iter * 4], &row_boxes[i * 4], iou_thresh);
                local_removed |= (iou_result << in_block);
            }
        }
    }
}

std::vector<int> nms_kernel(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold)
{
    int n = boxes.size();
    if (n <= 0) return {};
    
    // sort by score
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    // pack boxes into contiguous buffer
    std::vector<float> h_boxes(4 * n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4*i+0] = boxes[idx].x1;
        h_boxes[4*i+1] = boxes[idx].y1;
        h_boxes[4*i+2] = boxes[idx].x2;
        h_boxes[4*i+3] = boxes[idx].y2;
    }

    float* d_boxes = nullptr;
    bool* d_keep = nullptr;

    // memory setup
    cudaMalloc(&d_boxes, sizeof(float) * 4 * n);
    cudaMalloc(&d_keep,  sizeof(bool) * n);
    cudaMemcpy(d_boxes, h_boxes.data(), sizeof(float) * 4 * n, cudaMemcpyHostToDevice);
    cudaMemset(d_keep, 0, sizeof(bool) * n);


    // execute nms kernel
    auto total_start = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = ceil_div(n, MaxLoadPerThread);
    
    nms_kernel_1d<<<1, threadsPerBlock>>>(n, d_boxes, iou_threshold, d_keep);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    printf("=== NMS Kernel Overall Timing: %d microseconds ===", total_duration.count());

    // load results
    std::vector<uint8_t> h_keep(n, 0);
    cudaMemcpy(h_keep.data(), d_keep, sizeof(uint8_t) * n, cudaMemcpyDeviceToHost);

    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]){
            kept.push_back(i);
        }
    }

    cudaFree(d_boxes);
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
    printf("%d\n", result.size());
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
    
    if (nms_test(2000, 0)) {
        printf("CORRECT!!\n");
    }
    return 0;
}

// pixi run nvcc -arch=sm_86 -o "nms/nms-custom.out" "nms/nms-custom-v2.cu"

/*
Thoughtput:
nms_kernel_impl 36.16%
gather_keep_from_mask 2%

nms_kernel_1d 5%

Memory Read (KB):
org: 1
new: 173.23 KB

Memory Write (KB):
org: 20.99 KB + 0.512
new: 0.512

時間

*/