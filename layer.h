#pragma once

#include <algorithm>
#include <cassert>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "data.h"
#include "func.h"

namespace NeuralNetwork {
class Layer {
public:
    Layer() = default;

    Layer(int m, int n, ActFunc activation_func_name);

    void Read(std::ifstream& file);

    void Write(std::ofstream& file);

    void PrintFuncName(ActFunc func_name);


    Eigen::MatrixXd W;
    Eigen::MatrixXd B;
    Eigen::MatrixXd Z;

    Eigen::MatrixXd prevX;

    Eigen::MatrixXd nablaB;
    Eigen::MatrixXd nablaW;

    std::function<void(Eigen::MatrixXd&)> func;
    std::function<void(Eigen::MatrixXd&)> dx_func;

    ActFunc func_name;
};
}  // namespace NeuralNetwork