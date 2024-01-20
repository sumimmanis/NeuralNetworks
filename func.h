#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>

#include "Eigen/Dense"

namespace NeuralNetwork {
enum ActFunc { SIGMOID, RELU, TANH, SOFTMAX };

namespace impl {
void Sigmoid(Eigen::MatrixXd& X);

void DxSigmoid(Eigen::MatrixXd& X);

void Relu(Eigen::MatrixXd& X);

void DxRelu(Eigen::MatrixXd& X);

void Tanh(Eigen::MatrixXd& X);

void DxTanh(Eigen::MatrixXd& X);

void SoftMax(Eigen::MatrixXd& X);

void DxSoftMax(Eigen::MatrixXd& X);
}  // namespace impl

auto GetActFunc(ActFunc name) -> std::function<void(Eigen::MatrixXd&)>;

auto GetDxActFunc(ActFunc name) -> std::function<void(Eigen::MatrixXd&)>;
}  // namespace NeuralNetwork