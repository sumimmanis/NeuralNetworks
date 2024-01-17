#include <cassert>
#include <iostream>
#include <string>

#include "network.h"

class Ui {
public:
    Ui();

    void SetActFunc();

    void LoadAndPrime();

    void Prime();

    void Train();

    void SaveWithName();

    void Save();

    void CheckAccuracy();

private:
    ActFunc act_func_;
    NeuralNetwork network_;
    std::string in_name_;

    bool has_name_ = false;
    bool primed_ = false;
};
