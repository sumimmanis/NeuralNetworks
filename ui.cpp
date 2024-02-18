#include "ui.h"

namespace NeuralNetwork {
void Ui::LoadAndPrime() {
    try {
        std::cin >> in_name_;
        network_.LoadAndPrime("../data/params/" + in_name_);
        primed_ = true;
        has_name_ = true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::Prime() {
    std::deque<int> hidden_layers;
    std::deque<EnumActFunction> act_funcs;

    try {
        int num;
        std::cin >> num;
        for (int j = 0; j < num; ++j) {
            int layer;
            std::cin >> layer;
            hidden_layers.push_back(layer);
        }
        for (int j = 0; j < num; ++j) {
            std::string name;
            std::cin >> name;
            act_funcs.push_back(GetEnumActFunction(name));
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

    network_.Save("../data/params/" + out_name);
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

    network_.Save("../data/params/" + out_name);
    std::cout << "saved  " << out_name << std::endl;
}

void Ui::CheckAccuracy() {
    double error = network_.GetTestError();
    std::cout << "accuracy  " << error << std::endl;
}
}  // namespace NeuralNetwork