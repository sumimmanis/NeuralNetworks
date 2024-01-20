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
#include "util.h"
#include "layer.h"

namespace NeuralNetwork {

class NeuralNetwork {
public:
    NeuralNetwork() = default;

    void Prime(std::deque<int>& hidden_layers, std::deque<ActFunc>& func_names);

    void Train(int batch_size, double rate, int epoch);

    auto GetTestError() -> double;

    void Save(const std::string& name);

    void LoadAndPrime(const std::string& name);

private:
    void ForwardProp(Eigen::MatrixXd& X);

    void TrainRun(int batch_size, double rate, int num_of_batces,
                  std::vector<Eigen::MatrixXd>& vNbW, std::vector<Eigen::MatrixXd>& vNbB,
                  Eigen::MatrixXd& X);

    void LastError(Eigen::MatrixXd& X, Eigen::MatrixXd& Nb);

    void BackProp(Eigen::MatrixXd& Nb);

private:
    static constexpr int kTrainNum = 60000;
    static constexpr int kTestNum = 10000;

    static constexpr int kIn = 28 * 28;
    static constexpr int kOut = 10;

private:
    std::vector<Layer> layers_;
    DataLoader training_data_;
    int depth_;
};
}  // namespace NeuralNetwork
