#include "data.h"

namespace NeuralNetwork {
DataLoader::DataLoader(const std::string& name, int num, int size)
    : num_(num), size_(size), gen_(326) {
    std::ifstream file_digits;
    std::ifstream file_labels;

    file_digits = std::ifstream(name, std::ios::binary);
    file_labels = std::ifstream(name + "Label", std::ios::binary);

    assert(file_digits.is_open());
    assert(file_labels.is_open());

    digits_.resize(num * size);
    labels_.resize(num);

    pm_.resize(num);

    std::iota(pm_.begin(), pm_.end(), 0);

    file_digits.read(reinterpret_cast<char*>(digits_.data()), num * size);
    file_labels.read(reinterpret_cast<char*>(labels_.data()), num);

    file_digits.close();
    file_labels.close();
}

void DataLoader::SetMatrix(double* vec) {
    if (ind_ == num_) {
        ind_ = 0;
    }

    auto begin = digits_.begin() + pm_[ind_] * size_;
    std::copy(begin, begin + size_, vec);
}

void DataLoader::Randomise() {
    std::shuffle(pm_.begin(), pm_.end(), gen_);
}

auto DataLoader::GetLabel() -> int {
    return labels_[pm_[ind_++]];
}
}  // namespace NeuralNetwork