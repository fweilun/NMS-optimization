#include <vector>
#include <string>
#include <immintrin.h>
#include "utils.hpp"

// __m256 iou_SIMD(std::vector<float>& x1s, std::vector<float>& y1s, std::vector<float>& x2s, std::vector<float>& y2s, int i, int j);
struct Box { float x1, y1, x2, y2; };
std::vector<int> nms_baseline(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);
std::vector<int> nms_SoA(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);
std::vector<int> nms_omp(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);
std::vector<int> nms_SIMD(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);
