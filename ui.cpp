#include "ui.h"

Ui::Ui() : act_func_(ActFunc::SIGMOID) {}

void Ui::SetActFunc() {
    std::string arg;
    std::cin >> arg;
    if (arg == "SIGMOID") {
        act_func_ = ActFunc::SIGMOID;
    } else if (arg == "RELU") {
        act_func_ = ActFunc::RELU;
    } else if (arg == "TANH") {
        act_func_ = ActFunc::TANH;
    } else {
        std::cerr << "no such activation function, defaulted to SEGMOID" << std::endl;
    }

    if (primed_) {
        std::cerr << "the network has already been primed, your change will not take affect";
    }
}

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
    try {
        int num;
        std::cin >> num;
        for (int j = 0; j < num; ++j) {
            int layer;
            std::cin >> layer;
            hidden_layers.push_back(layer);
        }
        network_.Prime(hidden_layers, act_func_);
        primed_ = true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::Test() {
    try {
        if (not primed_) {
            throw std::invalid_argument("network is not primed");
        }
        int batch_size;
        double rate;
        int runs;
        std::cin >> batch_size >> rate >> runs;
        network_.Train(batch_size, rate, runs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void Ui::SaveWithName() {
    std::string out_name;
    std::cin >> out_name;
    double error = network_.GetTestError();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << error;

    network_.Save("../params/" + out_name);
    std::cout << "saved  " << out_name << std::endl;
}

void Ui::Save() {
    std::string out_name;
    double error = network_.GetTestError();
    std::cout << "error  " << error << std::endl;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << error;
    if (has_name_) {
        out_name = ss.str() + in_name_.substr(5);
    } else {
        out_name = ss.str();
    }

    network_.Save("../params/" + out_name);
    std::cout << "saved  " << out_name << std::endl;
}

void Ui::CheckAccuracy() {
    double error = network_.GetTestError();
    std::cout << "error  " << error << std::endl;
}