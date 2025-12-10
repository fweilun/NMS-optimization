#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <immintrin.h>
#include "nms.hpp"

__m256 iou_SIMD(std::vector<float>& x1s, std::vector<float>& y1s, std::vector<float>&x2s, std::vector<float>& y2s, int i, int j) {

    __m256 x1i = _mm256_set1_ps(x1s[i]);
    __m256 y1i = _mm256_set1_ps(y1s[i]);
    __m256 x2i = _mm256_set1_ps(x2s[i]);
    __m256 y2i = _mm256_set1_ps(y2s[i]);

    __m256 x1j = _mm256_loadu_ps(&x1s[j]);
    __m256 y1j = _mm256_loadu_ps(&y1s[j]);
    __m256 x2j = _mm256_loadu_ps(&x2s[j]);
    __m256 y2j = _mm256_loadu_ps(&y2s[j]);
    
    __m256 areaA = _mm256_mul_ps(_mm256_sub_ps(x2i, x1i), _mm256_sub_ps(y2i, y1i));
    __m256 areaB = _mm256_mul_ps(_mm256_sub_ps(x2j, x1j), _mm256_sub_ps(y2j, y1j));

    __m256 interx1 = _mm256_max_ps(x1i, x1j);
    __m256 intery1 = _mm256_max_ps(y1i, y1j);
    __m256 interx2 = _mm256_min_ps(x2i, x2j);
    __m256 intery2 = _mm256_min_ps(y2i, y2j);

    __m256 zero = _mm256_setzero_ps();
    __m256 w = _mm256_max_ps(zero, _mm256_sub_ps(interx2, interx1));
    __m256 h = _mm256_max_ps(zero, _mm256_sub_ps(intery2, intery1));
    __m256 inter = _mm256_mul_ps(w, h);
    __m256 iou_vec = _mm256_div_ps(inter, _mm256_sub_ps(_mm256_add_ps(areaA, areaB), inter));

    return iou_vec;
}

std::vector<int> nms_SIMD(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold) { 
    const int n = (int)boxes.size();
    if (n <= 0) return {};

    // 1) 按照 scores 排序（降序），就像 nms_kernel.cu 一样
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];  // 降序排序
    });

    // std::vector<float> h_boxes(4 * n);
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
    std::vector<int> h_keep(n, 1);

    for (int i = 0; i < n; ++i) {
        if (!h_keep[i]) continue;
        int j_end_simd = i + 1 + ((n - (i + 1)) / 8) * 8;
        #pragma omp parallel for
        for (int j_start = i + 1; j_start < n; j_start += 8) {
            __m256 iou_vec = iou_SIMD(x1s, y1s, x2s, y2s, i, j_start);

            float ious[8];
            _mm256_storeu_ps(ious, iou_vec); // 使用 storeu 進行非對齊儲存

            __m256 threshold_vec = _mm256_set1_ps(iou_threshold);
            
            __m256 mask = _mm256_cmp_ps(iou_vec, threshold_vec, _CMP_GT_OQ);
            
            int mask_int = _mm256_movemask_ps(mask);
            
            for (int k = 0; k < 8 && j_start + k < n; ++k) {
                if (mask_int & (1 << k)) {
                    int current_j = j_start + k;

                    #pragma omp atomic write
                    h_keep[current_j] = 0;
                }
            }
            for (int j = j_end_simd; j < n; ++j) {
                float areaA = (x2s[i] - x1s[i]) * (y2s[i] - y1s[i]);
                float areaB = (x2s[j] - x1s[j]) * (y2s[j] - y1s[j]);

                float ix1 = std::max(x1s[i], x1s[j]);
                float iy1 = std::max(y1s[i], y1s[j]);
                float ix2 = std::min(x2s[i], x2s[j]);
                float iy2 = std::min(y2s[i], y2s[j]);

                float w = std::max(0.0f, ix2 - ix1);
                float h = std::max(0.0f, iy2 - iy1);
                float inter = w * h;

                float iou = inter / (areaA + areaB - inter);
                if (iou > iou_threshold)
                    h_keep[j] = 0;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "=== NMS CPU Overall Timing: " << overall_duration.count() << " microseconds ===" << std::endl;

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