#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <time.h>
#include "nms.hpp"

const std::string BENCHMARK_DIR = "../benchmark/";

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
bool nms_test(int data_num, int id) {
    const std::string file_name = "rand_" + std::to_string(data_num) + "_" + std::to_string(id);
    const std::string data_path = BENCHMARK_DIR + file_name + ".csv";
    const std::string score_path = BENCHMARK_DIR + file_name + "_score.csv";
    const std::string answer_path = BENCHMARK_DIR + file_name + "_answer.csv";

    auto boxes_h = read_boxes_csv(data_path);
    auto scores_h = read_scores_csv(score_path);
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
    for (int id = 0; id < 10; id++) {
      if (nms_test(n, 0)) {
          printf("CORRECT!!\n");
      }
      else {
          printf("WRONG!!\n");
    }   
    }
    return 0;
}