#include "layer.h"

namespace NeuralNetwork {
Layer::Layer(int m, int n, EnumActFunction func_name) : func(GetActFunction(func_name)) {
    W = Eigen::MatrixXd(m, n);
    B = Eigen::MatrixXd(m, 1);
    W.setRandom();
    B.setRandom();
}

void Layer::ForwardProp(Eigen::MatrixXd& X) {
    prevX = X;
    Z = W * X + B;
    X = Z;
    func->Forward(X);
//    func->Forward(Z);
//    X = Z;
}

void Layer::BackProp(Eigen::MatrixXd& Nb, Eigen::MatrixXd& nextW) {
    func->Backward(Z);
    Nb = (nextW.transpose() * Nb).cwiseProduct(Z);

    nablaW = Nb * prevX.transpose();
    nablaB = Nb;
}

void Layer::Read(std::ifstream& file) {
    EnumActFunction func_name;
    file.read(reinterpret_cast<char*>(&func_name), sizeof(func_name));
    func = GetActFunction(func_name);

    std::cout << func->GetName() << std::endl;

    int m, n;
    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    W = Eigen::MatrixXd(m, n);
    file.read(reinterpret_cast<char*>(W.data()), sizeof(double) * W.size());

    std::cout << std::setw(3) << m << ' ' << std::setw(3) << n << std::endl;

    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    B = Eigen::MatrixXd(m, n);
    file.read(reinterpret_cast<char*>(B.data()), sizeof(double) * B.size());
}

void Layer::Write(std::ofstream& file) {
    EnumActFunction func_name = func->GetEnum();
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