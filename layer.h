#pragma once

#include <algorithm>
#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "data.h"
#include "func.h"

namespace NeuralNetwork {
class Layer {
public:
    Layer() = default;

    Layer(int m, int n, EnumActFunction func_name);

    void ForwardProp(Eigen::MatrixXd& X);

    void BackProp(Eigen::MatrixXd& Nb, Eigen::MatrixXd& nextW);

    void Read(std::ifstream& file);

    void Write(std::ofstream& file);

    std::unique_ptr<ActFunction> func;

    Eigen::MatrixXd W;
    Eigen::MatrixXd B;
    Eigen::MatrixXd Z;

    Eigen::MatrixXd prevX;

    Eigen::MatrixXd nablaB;
    Eigen::MatrixXd nablaW;
};
}  // namespace NeuralNetwork