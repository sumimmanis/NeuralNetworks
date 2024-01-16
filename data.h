#pragma once

#include <cassert>
#include <fstream>
#include <random>
#include <string>
#include <vector>

class Data {
public:
    Data() = default;

    Data(const std::string& name, int num, int size);

    void Fill(double* vec);

    auto GetLabel() -> int;

    void Randomise();

private:
    int num_;
    int size_;

    std::mt19937_64 gen_;

    std::vector<uint8_t> digits_;
    std::vector<uint8_t> labels_;

    std::vector<int> pm_;

    int ind_ = 0;
};
