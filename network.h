#pragma once

#include <algorithm>
#include <cassert>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "Extern/Eigen/Core"
#include "data.h"
#include "func.h"

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void Prime(std::deque<int>& hidden_layers, ActFunc act_func_name);

    void Train(int batch_size, double rate, int runs);

    auto GetTestError() -> double;

    void Save(const std::string& name);

    void LoadAndPrime(const std::string& name);

private:
    void InitRandomMatrix(Eigen::MatrixXd& mat, int m, int n);

    void ForwardProp(Eigen::MatrixXd& X);

    void TrainRun(int batch_size, double rate, int num_of_batces,
                  std::vector<Eigen::MatrixXd>& vNbW, std::vector<Eigen::MatrixXd>& vNbB,
                  Eigen::MatrixXd& X);

    void LastError(Eigen::MatrixXd& X, Eigen::MatrixXd& Nb);

    void BackProp(Eigen::MatrixXd& Nb);

private:
    struct Layer {
        Eigen::MatrixXd W;
        Eigen::MatrixXd B;
        Eigen::MatrixXd Z;

        Eigen::MatrixXd prevX;

        Eigen::MatrixXd nablB;
        Eigen::MatrixXd nablW;
    };

    static constexpr int kTtrainNum = 60000;
    static constexpr int kTestNum = 10000;

    static constexpr int kIn = 28 * 28;
    static constexpr int kOut = 10;

private:
    std::vector<Layer> layers_;

    Data train_data_;
    ActFunc func_name_;

    std::function<double(double)> func_;
    std::function<double(double)> dfunc_;

    int deapth_;
};
