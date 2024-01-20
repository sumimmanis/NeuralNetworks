#include "ui.h"

namespace NeuralNetwork {
void Ui::LoadAndPrime() {
    try {
        std::cin >> in_name_;
        network_.LoadAndPrime("../params/" + in_name_);
        primed_ = true;
        has_name_ = true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::Prime() {
    std::deque<int> hidden_layers;
    std::deque<ActFunc> act_funcs;

    try {
        int num;
        std::cin >> num;
        for (int j = 0; j < num; ++j) {
            int layer;
            std::cin >> layer;
            hidden_layers.push_back(layer);
        }
        for (int j = 0; j < num; ++j) {
            act_funcs.push_back(GetActFunc());
        }
        network_.Prime(hidden_layers, act_funcs);
        primed_ = true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::Train() {
    try {
        if (not primed_) {
            throw std::invalid_argument("network is not primed");
        }
        int batch_size;
        double rate;
        int epoch;
        std::cin >> batch_size >> rate >> epoch;
        network_.Train(batch_size, rate, epoch);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::SaveWithName() {
    std::string out_name;
    std::cin >> out_name;
    double error = network_.GetTestError();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << error;

    network_.Save("../params/" + out_name);
    std::cout << "saved  " << out_name << std::endl;
}

void Ui::Save() {
    std::string out_name;
    double error = network_.GetTestError();
    std::cout << "error  " << error << std::endl;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << error;
    if (has_name_) {
        out_name = ss.str() + in_name_.substr(6);
    } else {
        out_name = ss.str();
    }

    network_.Save("../params/" + out_name);
    std::cout << "saved  " << out_name << std::endl;
}

void Ui::CheckAccuracy() {
    double error = network_.GetTestError();
    std::cout << "accuracy  " << error << std::endl;
}

ActFunc Ui::GetActFunc() {
    std::string arg;
    std::cin >> arg;
    if (arg == "SIGMOID") {
        return SIGMOID;
    } else if (arg == "RELU") {
        return RELU;
    } else if (arg == "TANH") {
        return TANH;
    } else if (arg == "SOFTMAX" ) {
        return SOFTMAX;
    }
    std::cerr << "no such activation function, defaulted to SIGMOID" << std::endl;
    return SIGMOID;
}
}  // namespace NeuralNetwork