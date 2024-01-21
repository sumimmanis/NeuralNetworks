#include "util.h"

namespace NeuralNetwork {
auto IsCorrect(int size, double* vec, int label) -> bool {
    auto max_iter = std::max_element(vec, vec + size);

    return std::distance(vec, max_iter) == label;
}

void DxError(int size, double* vec, int label) {
    vec[label] = vec[label] - 1;
}

}  // namespace NeuralNetwork