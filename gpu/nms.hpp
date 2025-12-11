#include <vector>
#include <string>
#include <immintrin.h>

struct Box { float x1, y1, x2, y2; };

std::vector<int> nms_kernel(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);
