#pragma once

#include <algorithm>

namespace NeuralNetwork {
auto IsCorrect(int size, double* vec, int label) -> bool;

void DxError(int size, double* vec, int label);
}  // namespace NeuralNetwork