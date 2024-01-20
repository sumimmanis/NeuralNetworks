#pragma once

#include <cassert>
#include <iostream>
#include <string>

#include "network.h"

namespace NeuralNetwork {
class Ui {
public:
    Ui() = default;

    void LoadAndPrime();

    void Prime();

    void Train();

    void SaveWithName();

    void Save();

    void CheckAccuracy();

private:
    ActFunc GetActFunc();

private:
    NeuralNetwork network_;
    std::string in_name_;

    bool has_name_ = false;
    bool primed_ = false;
};
}
