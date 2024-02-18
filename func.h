#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "Eigen/Dense"

namespace NeuralNetwork {

enum EnumActFunction { SIGMOID, RELU, TANH, SOFTMAX };

class ActFunction {
public:
    virtual auto GetName() -> std::string = 0;
    virtual auto GetEnum() -> EnumActFunction = 0;

    virtual void Forward(Eigen::MatrixXd& X) = 0;
    virtual void Backward(Eigen::MatrixXd& X) = 0;

    virtual ~ActFunction() = default;
};

class Sigmoid : public ActFunction {
public:
    auto GetName() -> std::string override;
    auto GetEnum() -> EnumActFunction override;

    void Forward(Eigen::MatrixXd& X) override;
    void Backward(Eigen::MatrixXd& X) override;
};

class Relu : public ActFunction {
public:
    auto GetName() -> std::string override;
    auto GetEnum() -> EnumActFunction override;

    void Forward(Eigen::MatrixXd& X) override;
    void Backward(Eigen::MatrixXd& X) override;
};

class Tanh : public ActFunction {
public:
    auto GetName() -> std::string override;
    auto GetEnum() -> EnumActFunction override;

    void Forward(Eigen::MatrixXd& X) override;
    void Backward(Eigen::MatrixXd& X) override;
};

class SoftMax : public ActFunction {
public:
    auto GetName() -> std::string override;
    auto GetEnum() -> EnumActFunction override;

    void Forward(Eigen::MatrixXd& X) override;
    void Backward(Eigen::MatrixXd& X) override;
};

auto GetActFunction(EnumActFunction name) -> std::unique_ptr<ActFunction>;
auto GetEnumActFunction(const std::string& name) -> EnumActFunction;

}  // namespace NeuralNetwork
