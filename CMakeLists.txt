cmake_minimum_required(VERSION 3.22.1)

project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Try to find Eigen3 locally
find_package(Eigen3 QUIET)

if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIR}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    # If Eigen is not found, add it as a submodule
    message(STATUS "Eigen3 not found. Adding as a submodule.")
    execute_process(COMMAND git submodule update --init -- third-party/eigen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # Add Eigen as a submodule
    add_subdirectory(third-party/eigen)
    include_directories(third-party/eigen)
endif()

add_executable(${PROJECT_NAME} main.cpp network.cpp data.cpp func.cpp ui.cpp layer.cpp util.cpp)
