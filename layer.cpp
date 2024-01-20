#include "layer.h"

namespace NeuralNetwork {
Layer::Layer(int m, int n, ActFunc activation_func_name) {
    W = Eigen::MatrixXd(m, n);
    B = Eigen::MatrixXd(m, 1);
    W.setRandom();
    B.setRandom();

    func = GetActFunc(activation_func_name);
    dx_func = GetDxActFunc(activation_func_name);
    func_name = activation_func_name;
}

void Layer::PrintFuncName(ActFunc func_name) {
    switch (func_name) {
        case SIGMOID:
            std::cout << "SIGMOID" << std::endl;
            break;
        case RELU:
            std::cout << "RELU" << std::endl;
            break;
        case TANH:
            std::cout << "TANH" << std::endl;
            break;
        case SOFTMAX:
            std::cout << "SOFTMAX" << std::endl;
            break;
        default:
            assert(false);
    }
}

void Layer::Read(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&func_name), sizeof(func_name));
    PrintFuncName(func_name);
    func = GetActFunc(func_name);
    dx_func = GetDxActFunc(func_name);

    int m, n;
    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    file.read(reinterpret_cast<char*>(&n), sizeof(int));

    std::cout << std::setw(3) << m << ' ' << std::setw(3) << n << std::endl;
    W = Eigen::MatrixXd(m, n);
    file.read(reinterpret_cast<char*>(W.data()), sizeof(double) * W.size());

    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    B = Eigen::MatrixXd(m, n);
    file.read(reinterpret_cast<char*>(B.data()), sizeof(double) * B.size());
}

void Layer::Write(std::ofstream& file) {
    file.write(reinterpret_cast<char*>(&func_name), sizeof(func_name));
    int m = W.rows();
    int n = W.cols();
    file.write(reinterpret_cast<char*>(&m), sizeof(int));
    file.write(reinterpret_cast<char*>(&n), sizeof(int));
    file.write(reinterpret_cast<char*>(W.data()), sizeof(double) * W.size());
    m = B.rows();
    n = B.cols();
    file.write(reinterpret_cast<char*>(&m), sizeof(int));
    file.write(reinterpret_cast<char*>(&n), sizeof(int));
    file.write(reinterpret_cast<char*>(B.data()), sizeof(double) * B.size());
}
}  // namespace NeuralNetwork