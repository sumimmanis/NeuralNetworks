# neural network

## build with

```sh
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=DEBUG ..  
make NeuralNetworks     
```

## run

Prime the network at start with `-p` or `-lp` and then you can train, test, and save it.

`-lp name` loads file  `name` and primes it with loaded params.

`-p num layers...` primes the network. Set `num` of inner layers and specify their dimentions (one for each layer) with `layers...`

`-a act_func` set `act_func`, needs to be done before priming. Default is "SIGMOID", "RELU" and "TANH" are also available.

`-e` run tests ("error 0.990%" means that there was one mistake out of a hundred tests)

`-t batch_size rate runs`  trains the network

`-s` run tests. Save as file in "params/" with the `name` being the success rate. If the network was loaded as a file the name will be concatenated with the original file name's postfix.

`-S name` run tests. Save asfile in "params/" with the `name`.

`-r` returns

## example

Load and save and test my best attempt:

```sh
./NeuralNetworks 
-lp 0.911-40-50
-s
-r
```

outputs:

```sh
-lp 0.911-40-50
activation function name:  SIGMOID
dimentions of matrices M:
    40 784
    50  40
    10  50
-s
error  0.9109
saved  0.911-40-50
-r
```

## attempts

In "/params" I have some of my attempts at  trining. If there is a corresponding txt file than the network was trained once and in the txt file there are `batch_size rate runs` and `layers...` If there is no txt file than I've probably trained the network multiple times... but at least I've put `layers...` in the name xD.

## cppcheck

```sh
mkdir cppcheck-build

cppcheck --cppcheck-build-dir=cppcheck-build --std=c++20 --language=c++ --enable=all --suppressions-list=cppcheck-supressions.txt <file>
```

Does not work in main for cppcheck is too tired after checking the entire Eigen library... or maybe there are too much suppressed errors, I am still not sure
