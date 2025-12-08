#include "nms.hpp"
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <omp.h>

using namespace std;

float iou(std::vector<float>& h_boxes, int i, int j)
{
    // The format of box is [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    const float eps = 1e-6;
    float iou = 0.f;
    float areaA = (h_boxes[4 * i + 2] - h_boxes[4 * i + 0]) * (h_boxes[4 * i + 3] - h_boxes[4 * i + 1]);
    float areaB = (h_boxes[4 * j + 2] - h_boxes[4 * j + 0]) * (h_boxes[4 * j + 3] - h_boxes[4 * j + 1]);
    float x1 = max(h_boxes[4 * i + 0], h_boxes[4 * j + 0]);
    float y1 = max(h_boxes[4 * i + 1], h_boxes[4 * j + 1]);
    float x2 = max(h_boxes[4 * i + 2], h_boxes[4 * j + 2]);
    float y2 = max(h_boxes[4 * i + 3], h_boxes[4 * j + 3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    iou = inter / (areaA + areaB - inter + eps);
    return iou;
}

// std::vector<int> result = nms(boxes_h, scores_h, iou_thr);
std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold) {
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
    std::vector<unsigned char> h_keep(n, 1);

    for (int i = 0; i < n; ++i) {
        if (!h_keep[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (iou(h_boxes, i, j) > iou_threshold) h_keep[j] = 0;
        }
    }
    std::vector<int> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (h_keep[i]) {
            kept.push_back(order[i]);  // 返回原始索引
        }
    }
    // save_kept_to_csv(kept, "tempsave_standard.csv");
    return kept;
}

// 記得討論locality的問題!!
// double to float
