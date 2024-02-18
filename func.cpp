#include "func.h"

namespace NeuralNetwork {

auto Sigmoid::GetName() -> std::string {
    return "SIGMOID";
}

auto Sigmoid::GetEnum() -> EnumActFunction {
    return SIGMOID;
}

void Sigmoid::Forward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

void Sigmoid::Backward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) {
        double sigmoid = 1.0 / (1.0 + std::exp(-x));
        return sigmoid * (1.0 - sigmoid);
    });
}

auto Relu::GetName() -> std::string {
    return "RELU";
}

auto Relu::GetEnum() -> EnumActFunction {
    return RELU;
}

void Relu::Forward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return x > 0 ? x : 0.0; });
}

void Relu::Backward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [](double x) { return (x > 0.0) ? 1.0 : 0.0; });
}

auto Tanh::GetName() -> std::string {
    return "TANH";
}

auto Tanh::GetEnum() -> EnumActFunction {
    return TANH;
}

void Tanh::Forward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) { return std::tanh(x); });
}

void Tanh::Backward(Eigen::MatrixXd& X) {
    std::transform(X.data(), X.data() + X.size(), X.data(), [](double x) {
        double tanh_val = std::tanh(x);
        return 1.0 - tanh_val * tanh_val;
    });
}

auto SoftMax::GetName() -> std::string {
    return "SOFTMAX";
}

auto SoftMax::GetEnum() -> EnumActFunction {
    return SOFTMAX;
}

void SoftMax::Forward(Eigen::MatrixXd& X) {
    double norm = 1 / std::transform_reduce(X.data(), X.data() + X.size(), 0.0, std::plus<>(),
                                            [](double x) { return std::exp(x); });

    std::transform(X.data(), X.data() + X.size(), X.data(),
                   [norm](double x) { return std::exp(x) * norm; });
}

void SoftMax::Backward(Eigen::MatrixXd& X) {
    //    Eigen::MatrixXd J(X.rows(), X.rows());
    assert(false);
}

auto GetActFunction(EnumActFunction name) -> std::unique_ptr<ActFunction> {
    switch (name) {
        case SIGMOID:
            return std::make_unique<Sigmoid>();
        case RELU:
            return std::make_unique<Relu>();
        case TANH:
            return std::make_unique<Tanh>();
        case SOFTMAX:
            return std::make_unique<SoftMax>();
        default:
            assert(false);
    }
}

auto GetEnumActFunction(const std::string& name) -> EnumActFunction {
    if (name == "SIGMOID") {
        return SIGMOID;
    }
    if (name == "RELU") {
        return RELU;
    }
    if (name == "TANH") {
        return TANH;
    }
    if (name == "SOFTMAX") {
        return SOFTMAX;
    }
    throw std::invalid_argument("no such activation function");
}

}  // namespace NeuralNetwork
