#include "func.h"

namespace NeuralNetwork {
namespace impl {
void Sigmoid(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

void DxSigmoid(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) {
        double sigmoid = 1.0 / (1.0 + std::exp(-x));
        return sigmoid * (1.0 - sigmoid);
    });
}

void Relu(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return x > 0 ? x : 0.0; });
}

void DxRelu(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return (x > 0.0) ? 1.0 : 0.0; });
}

void Tanh(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) { return std::tanh(x); });
}

void DxTanh(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) {
        double tanh_val = std::tanh(x);
        return 1.0 - tanh_val * tanh_val;
    });
}

void SoftMax(Eigen::MatrixXd& X) {
    double norm = 1 / std::transform_reduce(X.data(), X.data() + X.size(), 0.0, std::plus<>(),
                                            [](double x) { return std::exp(x); });

    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [norm](double x) { return std::exp(x) * norm; });
}

void DxSoftMax(Eigen::MatrixXd& X) {
    assert(false);
}
}  // namespace impl

auto GetActFunc(ActFunc name) -> std::function<void(Eigen::MatrixXd&)> {
    switch (name) {
        case SIGMOID:
            return impl::Sigmoid;
        case RELU:
            return impl::Relu;
        case TANH:
            return impl::Tanh;
        case SOFTMAX:
            return impl::SoftMax;
        default:
            assert(false);
    }
}

auto GetDxActFunc(ActFunc name) -> std::function<void(Eigen::MatrixXd&)> {
    switch (name) {
        case SIGMOID:
            return impl::DxSigmoid;
        case RELU:
            return impl::DxRelu;
        case TANH:
            return impl::DxTanh;
        case SOFTMAX:
            return impl::DxSoftMax;
        default:
            assert(false);
    }
}
}  // namespace NeuralNetwork
