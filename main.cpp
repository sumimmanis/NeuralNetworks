#include "ui.h"

auto main() -> int {
    NeuralNetwork::Ui ui;

    while (true) {
        std::string arg;
        std::cin >> arg;

        if (arg == "-lp") {
            ui.LoadAndPrime();
        } else if (arg == "-p") {
            ui.Prime();
        } else if (arg == "-t") {
            ui.Train();
        } else if (arg == "-S") {
            ui.SaveWithName();
        } else if (arg == "-s") {
            ui.Save();
        } else if (arg == "-e") {
            ui.CheckAccuracy();
        } else if (arg == "-r") {
            return 0;
        }
    }
}
