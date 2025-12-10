#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

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
    for (int idx : kept)
        file << idx << "\n";

    file.close();
    std::cout << "Saved " << kept.size() << " indices to " << filename << std::endl;
}

float iou(std::vector<float>& h_boxes, int i, int j)
{
    // The format of box is [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    // const float eps = 1e-6;
    float iou = 0.f;
    float areaA = (h_boxes[4 * i + 2] - h_boxes[4 * i + 0]) * (h_boxes[4 * i + 3] - h_boxes[4 * i + 1]);
    float areaB = (h_boxes[4 * j + 2] - h_boxes[4 * j + 0]) * (h_boxes[4 * j + 3] - h_boxes[4 * j + 1]);
    float x1 = std::max(h_boxes[4 * i + 0], h_boxes[4 * j + 0]);
    float y1 = std::max(h_boxes[4 * i + 1], h_boxes[4 * j + 1]);
    float x2 = std::min(h_boxes[4 * i + 2], h_boxes[4 * j + 2]);
    float y2 = std::min(h_boxes[4 * i + 3], h_boxes[4 * j + 3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    iou = inter / (areaA + areaB - inter);
    return iou;
}