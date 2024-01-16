#include "network.h"

void NeuralNetwork::Prime(std::deque<int>& hidden_layers, ActFunc act_func_name) {
    func_ = GetActFunc(act_func_name);
    dfunc_ = GetDxActFunc(act_func_name);

    deapth_ = hidden_layers.size() + 1;

    hidden_layers.push_front(kIn);
    hidden_layers.push_back(kOut);

    layers_.reserve(deapth_);

    for (int i = 0; i < deapth_; ++i) {
        int m = hidden_layers[i + 1];
        int n = hidden_layers[i];

        InitRandomMatrix(layers_[i].W, m, n);
    }

    for (int i = 0; i < deapth_; ++i) {
        int m = hidden_layers[i + 1];

        InitRandomMatrix(layers_[i].B, m, 1);
    }

    train_data_ = Data("../Extern/MNIST/parced_60k", kTtrainNum, kIn);
    func_name_ = act_func_name;
}

void NeuralNetwork::Train(int batch_size, double rate, int runs) {
    std::vector<Eigen::MatrixXd> vNbW(deapth_);
    std::vector<Eigen::MatrixXd> vNbB(deapth_);

    for (int i = 0; i < deapth_; ++i) {
        vNbW[i] = Eigen::MatrixXd(layers_[i].W.rows(), layers_[i].W.cols());
        vNbB[i] = Eigen::MatrixXd(layers_[i].B.rows(), layers_[i].B.cols());
    }

    for (int run = 0; run < runs; ++run) {
        std::cout << "run  " << run << "  out of  " << runs << " ... " << std::flush;

        TrainRun(batch_size, rate, vNbW, vNbB);
        train_data_.Randomise();

        std::cout << std::endl;
    }
}

void NeuralNetwork::TrainRun(int batch_size, int rate, std::vector<Eigen::MatrixXd>& vNbW,
                             std::vector<Eigen::MatrixXd>& vNbB) {
    Eigen::MatrixXd X;
    int num_of_batces = kTtrainNum / batch_size;
    for (int i = 0; i < num_of_batces; i++) {
        for (int j = 0; j < deapth_; ++j) {
            vNbW[j].setZero();
            vNbB[j].setZero();
        }

        for (int batch = 0; batch < batch_size; ++batch) {
            X.resize(kIn, 1);

            train_data_.Fill(X.data());
            ForwardProp(X);

            Eigen::MatrixXd Nb;
            LastError(X, Nb);
            BackProp(Nb);

            for (int j = 0; j < deapth_; ++j) {
                vNbW[j] += layers_[j].nablW;
                vNbB[j] += layers_[j].nablB;
            }
        }

        for (int j = 0; j < deapth_; ++j) {
            layers_[j].W -= rate / num_of_batces * vNbW[j];
            layers_[j].B -= rate / num_of_batces * vNbB[j];
        }
    }
}


void NeuralNetwork::InitRandomMatrix(Eigen::MatrixXd& mat, int m, int n) {
    Eigen::MatrixXd matrix(m, n);

    matrix.setRandom();

#ifdef POSITIVE_RANDOM
    std::for_each(matrix.data(), matrix.data() + matrix.size(),
                  [](double& a) { a = (a + 1) * 0.5; });
#endif

    mat = matrix;
}

void NeuralNetwork::ForwardProp(Eigen::MatrixXd& X) {
    for (int i = 0; i < deapth_; ++i) {
        layers_[i].prevX = X;

        layers_[i].Z = layers_[i].W * X + layers_[i].B;
        X = layers_[i].Z.unaryExpr(func_);
    }
}

void NeuralNetwork::LastError(Eigen::MatrixXd& X, Eigen::MatrixXd& Nb) {
    int y = train_data_.GetLabel();

    DxErrorInplace(kOut, X.data(), y);

    auto& CurrLayer = layers_[deapth_ - 1];

    Nb = X.cwiseProduct(CurrLayer.Z.unaryExpr(dfunc_));

    CurrLayer.nablB = Nb;
    CurrLayer.nablW = Nb * CurrLayer.prevX.transpose();
}


void NeuralNetwork::BackProp(Eigen::MatrixXd& Nb) {
    for (int i = deapth_ - 1; i > 0; --i) {
        auto& CurrLayer = layers_[i - 1];

        Nb = (layers_[i].W.transpose() * Nb).cwiseProduct(CurrLayer.Z.unaryExpr(dfunc_));

        CurrLayer.nablB = Nb;
        CurrLayer.nablW = Nb * CurrLayer.prevX.transpose();
    }
}

void NeuralNetwork::Save(const std::string& name) {
    std::ofstream file;

    file.open(name, std::ios::out);

    file.write(reinterpret_cast<char*>(&deapth_), sizeof(int));
    file.write(reinterpret_cast<char*>(&func_name_), sizeof(func_name_));

    for (int i = 0; i < deapth_; ++i) {
        int m = layers_[i].W.rows();
        int n = layers_[i].W.cols();
        file.write(reinterpret_cast<char*>(&m), sizeof(int));
        file.write(reinterpret_cast<char*>(&n), sizeof(int));
        file.write(reinterpret_cast<char*>(layers_[i].W.data()),
                   sizeof(double) * layers_[i].W.size());
    }

    for (int i = 0; i < deapth_; ++i) {
        int m = layers_[i].B.rows();
        int n = layers_[i].B.cols();
        file.write(reinterpret_cast<char*>(&m), sizeof(int));
        file.write(reinterpret_cast<char*>(&n), sizeof(int));
        file.write(reinterpret_cast<char*>(layers_[i].B.data()),
                   sizeof(double) * layers_[i].B.size());
    }
    file.close();
}

void NeuralNetwork::LoadAndPrime(const std::string& name) {
    std::ifstream file;

    file.open(name, std::ios::in);

    if (not file.is_open()) {
        throw std::invalid_argument("file not found, try again");
    }

    file.read(reinterpret_cast<char*>(&deapth_), sizeof(int));
    file.read(reinterpret_cast<char*>(&func_name_), sizeof(func_name_));

    func_ = GetActFunc(func_name_);
    dfunc_ = GetDxActFunc(func_name_);

    layers_.resize(deapth_);


    switch (func_name_) {
        case ActFunc::SIGMOID:
            std::cout << "activation function name:  "
                      << "SIGMOID" << std::endl;
            break;
        case ActFunc::RELU:
            std::cout << "activation function name:  "
                      << "RELU" << std::endl;
            break;

        case ActFunc::TANH:
            std::cout << "activation function name:  "
                      << "RELU" << std::endl;
            break;
        default:
            assert(false);
    }

    std::cout << "dimentions of matrices M:" << std::endl;

    for (int i = 0; i < deapth_; ++i) {
        int m, n;
        file.read(reinterpret_cast<char*>(&m), sizeof(int));
        file.read(reinterpret_cast<char*>(&n), sizeof(int));

        std::cout << '\t' << std::setw(3) << m << ' ' << std::setw(3) << n << std::endl;
        layers_[i].W = Eigen::MatrixXd(m, n);
        file.read(reinterpret_cast<char*>(layers_[i].W.data()),
                  sizeof(double) * layers_[i].W.size());
    }

    for (int i = 0; i < deapth_; ++i) {
        int m, n;
        file.read(reinterpret_cast<char*>(&m), sizeof(int));
        file.read(reinterpret_cast<char*>(&n), sizeof(int));
        layers_[i].B = Eigen::MatrixXd(m, n);
        file.read(reinterpret_cast<char*>(layers_[i].B.data()),
                  sizeof(double) * layers_[i].B.size());
    }

    file.close();

    train_data_ = Data("../Extern/MNIST/parced_60k", kTtrainNum, kIn);
}

auto NeuralNetwork::GetTestError() -> double {
    Data test_data("../Extern/MNIST/parced_10k", kTtrainNum, kIn);
    Eigen::MatrixXd X;
    double matches = 0;
    for (int i = 0; i < kTestNum; ++i) {
        X.resize(kIn, 1);

        test_data.Fill(X.data());
        ForwardProp(X);

        int y = test_data.GetLabel();

        auto max_iter = std::max_element(X.data(), X.data() + X.size());

        if (std::distance(X.data(), max_iter) == y) {
            matches += 1;
        }
    }

    return matches / kTestNum;
}
