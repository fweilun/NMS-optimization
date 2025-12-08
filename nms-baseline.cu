#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>


const std::string BENCHMARK_DIR = "./benchmark/";

struct Box { float x1, y1, x2, y2; };

// define cuda check function
void checkCuda() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compute_overlap_matrix launch error: "
                << cudaGetErrorString(err) << "\n";
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "compute_overlap_matrix runtime error: "
                << cudaGetErrorString(err) << "\n";
    }
}

// ======================== CSV READ/WRITE UTILITIES ========================

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

// ======================== SIMPLE CUDA NMS IMPLEMENTATION ========================

// conceptual 2D indexing into a flat n×n array
#define OVERLAP(mat, n, i, j) (mat[(i) * (n) + (j)])

// Device IoU between box i and j, using packed [x1,y1,x2,y2] array
__device__ float devIoU_single(const float* boxes, int i, int j)
{
    const float* a = boxes + 4 * i;
    const float* b = boxes + 4 * j;

    float left   = fmaxf(a[0], b[0]);
    float top    = fmaxf(a[1], b[1]);
    float right  = fminf(a[2], b[2]);
    float bottom = fminf(a[3], b[3]);

    float w = fmaxf(right  - left,  0.0f);
    float h = fmaxf(bottom - top,   0.0f);
    float inter = w * h;

    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float uni = area_a + area_b - inter;

    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

// Kernel 1: compute upper-triangular IoU > thr matrix
__global__ void compute_overlap_matrix(
    int n,
    float iou_threshold,
    const float* boxes,   // [n * 4]
    bool* overlap)        // [n * n]
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // the jth box
    int i = blockIdx.y * blockDim.y + threadIdx.y; // the ith box

    if (i >= n || j >= n) return;
    if (i >= j) return; // only fill i < j (upper triangle)

    float iou = devIoU_single(boxes, i, j);
    OVERLAP(overlap, n, i, j) = (iou > iou_threshold); // results are stored in the pointer 'overlap'
}

// Kernel 2: greedy NMS on GPU using the overlap matrix
// Runs on a single thread for simplicity.
__global__ void greedy_nms_from_overlap(
    int n,
    const bool* overlap,  // [n * n], only i<j used
    bool* keep)           // [n]
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Now, 'keep' is still initialized to all false.

    for (int i = 0; i < n; ++i) {
        bool suppressed = false;

        // Check overlap with any earlier kept box
        for (int k = 0; k < i && !suppressed; ++k) {
            if (!keep[k]) continue;

            int row = (k < i) ? k : i;
            int col = (k < i) ? i : k;

            if (OVERLAP(overlap, n, row, col)) {
                suppressed = true;
            }
        }
        keep[i] = !suppressed;
    }
}

// Host-side NMS wrapper, same signature as your original nms_kernel
std::vector<int> nms_kernel(const std::vector<Box>& boxes,
                            const std::vector<float>& scores,
                            double iou_threshold)
{
    const int n = static_cast<int>(boxes.size());
    if (n <= 0) return {};

    // 1) sort indices by score (descending)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // 2) re-pack boxes in sorted order
    std::vector<float> h_boxes(4 * n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4 * i + 0] = boxes[idx].x1;
        h_boxes[4 * i + 1] = boxes[idx].y1;
        h_boxes[4 * i + 2] = boxes[idx].x2;
        h_boxes[4 * i + 3] = boxes[idx].y2;
    }

    // 3) allocate device buffers
    float* d_boxes   = nullptr;
    bool*  d_overlap = nullptr;
    bool*  d_keep    = nullptr;

    cudaMalloc(&d_boxes,   sizeof(float) * 4 * n);
    cudaMalloc(&d_overlap, sizeof(bool)  * n * n);
    cudaMalloc(&d_keep,    sizeof(bool)  * n);

    cudaMemcpy(d_boxes, h_boxes.data(),
               sizeof(float) * 4 * n, cudaMemcpyHostToDevice);
    cudaMemset(d_overlap, 0, sizeof(bool) * n * n);
    cudaMemset(d_keep,    0, sizeof(bool) * n);

    // 4) kernel 1: compute overlap matrix
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    auto compute_overlap_start = std::chrono::high_resolution_clock::now();

    compute_overlap_matrix<<<grid, block>>>(
        n,
        static_cast<float>(iou_threshold),
        d_boxes,
        d_overlap
    );
    cudaDeviceSynchronize();
    checkCuda();

    auto compute_overlap_end = std::chrono::high_resolution_clock::now();
    auto compute_overlap_duration = std::chrono::duration_cast<std::chrono::microseconds>(compute_overlap_end - compute_overlap_start);
    std::cout << "--- compute_overlap time: " << compute_overlap_duration.count() << " microseconds ---" << std::endl;

    auto greedy_nms_start = std::chrono::high_resolution_clock::now();

    // 5) kernel 2: greedy NMS (single thread)
    greedy_nms_from_overlap<<<1, 1>>>(n, d_overlap, d_keep);
    cudaDeviceSynchronize();
    checkCuda();

    auto greedy_nms_end = std::chrono::high_resolution_clock::now();
    auto greedy_nms_duration = std::chrono::duration_cast<std::chrono::microseconds>(greedy_nms_end - greedy_nms_start);
    std::cout << "--- greedy_nms time: " << greedy_nms_duration.count() << " microseconds ---" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "=== Total NMS time: " << total_duration.count() << " microseconds ===" << std::endl;
    
    // 6) copy keep flags back
    std::vector<unsigned char> h_keep(n, 0);
    cudaMemcpy(h_keep.data(), d_keep,
               sizeof(bool) * n, cudaMemcpyDeviceToHost);

    // 7) build kept indices in terms of original order
    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]) {
            kept.push_back(order[i]);  // map from sorted index to original index
        }
    }

    // Optionally save to CSV just like your original code
    save_kept_to_csv(kept, "tempsave_simple.csv");

    // 8) free device memory
    cudaFree(d_boxes);
    cudaFree(d_overlap);
    cudaFree(d_keep);

    return kept;
}

// ======================== TEST & MAIN ========================

bool nms_test(int data_num, int id) {
    const std::string file_name = "rand_" + std::to_string(data_num) + "_" + std::to_string(id);
    const std::string data_path = BENCHMARK_DIR + file_name + ".csv";
    const std::string score_path = BENCHMARK_DIR + file_name + "_score.csv";
    const std::string answer_path = BENCHMARK_DIR + file_name + "_answer.csv";

    auto boxes_h = read_boxes_csv(data_path);
    auto scores_h = read_scores_csv(score_path);

    std::cout << "boxes_h.size() = " << boxes_h.size()
          << ", scores_h.size() = " << scores_h.size() << "\n";

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
    else {
        printf("WRONG!!\n");
    }
    return 0;
}