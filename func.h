#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>

enum class ActFunc { SIGMOID, RELU, TANH };

auto Sigmoid(double x) -> double;

auto DxSigmoid(double x) -> double;

auto Relu(double x) -> double;

auto DxRelu(double x) -> double;

auto Tanh(double x) -> double;

auto DxTanh(double x) -> double;

auto GetActFunc(ActFunc name) -> std::function<double(double)>;

auto GetDxActFunc(ActFunc name) -> std::function<double(double)>;

void SoftMaxInplace(int size, double* vec);

auto IsCorrectResult(int size, double* vec, int label) -> bool;

void DxErrorInplace(int size, double* vec, int label);
