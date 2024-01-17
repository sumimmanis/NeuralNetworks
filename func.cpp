#include "func.h"

auto Sigmoid(double x) -> double {
    return 1.0 / (1.0 + std::exp(-x));
}

auto DxSigmoid(double x) -> double {
    return Sigmoid(x) * (1 - Sigmoid(x));
}

auto Relu(double x) -> double {
    return std::max(0.0, x);
}

auto DxRelu(double x) -> double {
    return (x > 0) ? 1.0 : 0.0;
}

auto Tanh(double x) -> double {
    return std::tanh(x);
}

auto DxTanh(double x) -> double {
    return 1.0 - std::tanh(x) * std::tanh(x);
}


auto GetActFunc(ActFunc name) -> std::function<double(double)> {
    switch (name) {
        case ActFunc::SIGMOID:
            return Sigmoid;
        case ActFunc::RELU:
            return Relu;
        case ActFunc::TANH:
            return Tanh;
        default:
            assert(false);
    }
}

auto GetDxActFunc(ActFunc name) -> std::function<double(double)> {
    switch (name) {
        case ActFunc::SIGMOID:
            return DxSigmoid;
        case ActFunc::RELU:
            return DxRelu;
        case ActFunc::TANH:
            return DxTanh;
        default:
            assert(false);
    }
}

void SoftMaxInplace(int size, double* vec) {
    double norm = 1 / std::transform_reduce(vec, vec + size, 0.0, std::plus<>(),
                                            [](double x) { return std::exp(x); });

    std::transform(vec, vec + size, vec, [norm](double x) { return std::exp(x) * norm; });
}

auto IsCorrectResult(int size, double* vec, int label) -> bool {
    auto max_iter = std::max_element(vec, vec + size);

    return std::distance(vec, max_iter) == label;
}

void DxErrorInplace(int size, double* vec, int label) {
    vec[label] = vec[label] - 1;
}
