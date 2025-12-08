#include <vector>

struct Box { float x1, y1, x2, y2; };
std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores, double iou_threshold);

