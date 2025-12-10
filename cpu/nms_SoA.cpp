#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "nms.hpp"

float iou_SoA(std::vector<float>& x1s, std::vector<float>& y1s, std::vector<float>&x2s, std::vector<float>& y2s, int i, int j) {
    // The format of box is [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    // const float eps = 1e-6;
    float x1i = x1s[i];
    float y1i = y1s[i];
    float x2i = x2s[i];
    float y2i = y2s[i];

    float x1j = x1s[j];
    float y1j = y1s[j];
    float x2j = x2s[j];
    float y2j = y2s[j];
    
    float iou = 0.f;
    float areaA = (x2i - x1i) * (y2i - y1i);
    float areaB = (x2j - x1j) * (y2j - y1j);
    float inter_x1 = std::max(x1i, x1j);
    float inter_y1 = std::max(y1i, y1j);
    float inter_x2 = std::min(x2i, x2j);
    float inter_y2 = std::min(y2i, y2j);
    float w = std::max(0.f, inter_x2 - inter_x1);
    float h = std::max(0.f, inter_y2 - inter_y1);
    float inter = w * h;
    iou = inter / (areaA + areaB - inter);
    return iou;
}
std::vector<int> nms_SoA(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold) { 
    const int n = (int)boxes.size();
    if (n <= 0) return {};

    // 1) 按照 scores 排序（降序），就像 nms_kernel.cu 一样
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    std::vector<float> x1s(n);
    std::vector<float> y1s(n);
    std::vector<float> x2s(n);
    std::vector<float> y2s(n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        x1s[i] = boxes[idx].x1;
        y1s[i] = boxes[idx].y1;
        x2s[i] = boxes[idx].x2;
        y2s[i] = boxes[idx].y2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> h_keep(n, 1);
    
    for (int i = 0; i < n; ++i) {
        if (!h_keep[i]) continue;
        // #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            if (iou_SoA(x1s, y1s, x2s, y2s, i, j) > iou_threshold) h_keep[j] = 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "=== NMS CPU SoA Overall Timing: " << overall_duration.count() << " microseconds ===" << std::endl;

    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]) {
            kept.push_back(order[i]);  // 返回原始索引
        }
    }
    save_kept_to_csv(kept, "tempsave_standard.csv");
    return kept;
}