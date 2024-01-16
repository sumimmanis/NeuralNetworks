#include "ui.h"

auto main() -> int {
    Ui ui;

    while (true) {
        std::string arg;
        std::cin >> arg;

        if (arg == "-a") {
            ui.SetActFunc();
        } else if (arg == "-lp") {
            ui.LoadAndPrime();
        } else if (arg == "-p") {
            ui.Prime();
        } else if (arg == "-t") {
            ui.Test();
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
