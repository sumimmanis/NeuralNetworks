#pragma once

#include <cassert>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace NeuralNetwork {
class DataLoader {
public:
    DataLoader() = default;

    DataLoader(const std::string& name, int num, int size);

    void SetMatrix(double* vec);

    auto GetLabel() -> int;

    void Randomise();

private:
    int num_;
    int size_;

    std::mt19937_64 gen_;
    std::vector<int> pm_;

    std::vector<uint8_t> digits_;
    std::vector<uint8_t> labels_;

    int ind_ = 0;
};
}  // namespace NeuralNetwork