#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "nms.hpp"
#include "utils.hpp"

std::vector<int> nms_baseline(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold) { 
    const int n = (int)boxes.size();
    if (n <= 0) return {};

    // 1) 按照 scores 排序（降序），就像 nms_kernel.cu 一样
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    std::vector<float> h_boxes(4 * n);
    for (int i = 0; i < n; ++i) {
        int idx = order[i];
        h_boxes[4*i+0] = boxes[idx].x1;
        h_boxes[4*i+1] = boxes[idx].y1;
        h_boxes[4*i+2] = boxes[idx].x2;
        h_boxes[4*i+3] = boxes[idx].y2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<unsigned char> h_keep(n, 1);
    
    for (int i = 0; i < n; ++i) {
        if (!h_keep[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (iou(h_boxes, i, j) > iou_threshold) h_keep[j] = 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "=== NMS CPU Baseline Overall Timing: " << overall_duration.count() << " microseconds ===" << std::endl;

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


