cmake_minimum_required(VERSION 3.22.1)

project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

add_executable(${PROJECT_NAME} main.cpp network.cpp data.cpp func.cpp ui.cpp layer.cpp util.cpp)