#include <vector>
#include <string>


float iou(std::vector<float>& h_boxes, int i, int j);
void save_kept_to_csv(std::vector<int>& kept, const std::string& filename);