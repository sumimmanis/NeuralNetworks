#include "network.h"

namespace NeuralNetwork {


void NeuralNetwork::Prime(std::deque<int>& hidden_layers, std::deque<EnumActFunction>& func_names) {
    depth_ = hidden_layers.size() + 1;

    hidden_layers.push_front(kIn);
    hidden_layers.push_back(kOut);

    layers_.reserve(depth_);

    for (int i = 0; i < depth_; ++i) {
        int m = hidden_layers[i + 1];
        int n = hidden_layers[i];

        layers_[i] = Layer(m, n, func_names[i]);
    }

    training_data_ = DataLoader("../data/MNIST/parced_60k", kTrainNum, kIn);
}

void NeuralNetwork::Train(int batch_size, double rate, int epoch) {
    std::vector<Eigen::MatrixXd> vNbW(depth_);
    std::vector<Eigen::MatrixXd> vNbB(depth_);

    Eigen::MatrixXd X;
    int num_of_batches = kTrainNum / batch_size;

    for (int i = 0; i < depth_; ++i) {
        vNbW[i] = Eigen::MatrixXd(layers_[i].W.rows(), layers_[i].W.cols());
        vNbB[i] = Eigen::MatrixXd(layers_[i].B.rows(), layers_[i].B.cols());
    }

    for (int i = 0; i < epoch; ++i) {
        std::cout << "running  " << i << "  out of  " << epoch << std::endl;

        TrainRun(batch_size, rate, num_of_batches, vNbW, vNbB, X);
        training_data_.Shuffle();
    }
    std::cout << "done" << std::endl;
}

void NeuralNetwork::TrainRun(int batch_size, double rate, int num_of_batches,
                             std::vector<Eigen::MatrixXd>& vNbW, std::vector<Eigen::MatrixXd>& vNbB,
                             Eigen::MatrixXd& X) {

    for (int i = 0; i < num_of_batches; i++) {
        for (int j = 0; j < depth_; ++j) {
            vNbW[j].setZero();
            vNbB[j].setZero();
        }

        for (int batch = 0; batch < batch_size; ++batch) {
            X.resize(kIn, 1);

            training_data_.SetMatrix(X.data());
            ForwardProp(X);

            Eigen::MatrixXd Nb;
            LastError(X, Nb);
            BackProp(Nb);

            for (int j = 0; j < depth_; ++j) {
                vNbW[j] += layers_[j].nablaW;
                vNbB[j] += layers_[j].nablaB;
            }
        }

        for (int j = 0; j < depth_; ++j) {
            layers_[j].W -= rate / num_of_batches * vNbW[j];
            layers_[j].B -= rate / num_of_batches * vNbB[j];
        }
    }
}

void NeuralNetwork::ForwardProp(Eigen::MatrixXd& X) {
    for (int i = 0; i < depth_; ++i) {
        layers_[i].ForwardProp(X);
    }
}

void NeuralNetwork::LastError(Eigen::MatrixXd& X, Eigen::MatrixXd& Nb) {
    int y = training_data_.GetLabel();

    DxError(kOut, X.data(), y);

    auto& CurrLayer = layers_[depth_ - 1];

    CurrLayer.func->Backward(CurrLayer.Z);
    Nb = X.cwiseProduct(CurrLayer.Z);

    Nb = X;
    CurrLayer.nablaB = Nb;
    CurrLayer.nablaW = Nb * CurrLayer.prevX.transpose();
}


void NeuralNetwork::BackProp(Eigen::MatrixXd& Nb) {
    for (int i = depth_ - 1; i > 0; --i) {
        layers_[i - 1].BackProp(Nb, layers_[i].W);
    }
}

void NeuralNetwork::Save(const std::string& name) {
    std::ofstream file;
    file.open(name, std::ios::out);
    file.write(reinterpret_cast<char*>(&depth_), sizeof(int));
    for (int i = 0; i < depth_; ++i) {
        layers_[i].Write(file);
    }
    file.close();
}

void NeuralNetwork::LoadAndPrime(const std::string& name) {
    std::ifstream file;
    file.open(name, std::ios::in);
    if (not file.is_open()) {
        throw std::invalid_argument("file not found, try again");
    }
    file.read(reinterpret_cast<char*>(&depth_), sizeof(int));
    layers_.resize(depth_);
    for (int i = 0; i < depth_; ++i) {
        layers_[i].Read(file);
    }
    file.close();
    training_data_ = DataLoader("../data/MNIST/parced_60k", kTrainNum, kIn);
}

auto NeuralNetwork::GetTestError() -> double {
    DataLoader test_data("../data/MNIST/parced_10k", kTrainNum, kIn);
    Eigen::MatrixXd X;
    int matches = 0;
    for (int i = 0; i < kTestNum; ++i) {
        X.resize(kIn, 1);
        test_data.SetMatrix(X.data());
        ForwardProp(X);
        matches += IsCorrect(X.size(), X.data(), test_data.GetLabel());
    }
    return static_cast<double>(matches) / kTestNum;
}
}  // namespace NeuralNetwork
