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

`-p num layers...  func... ` primes the network. Set `num` of inner layers and specify their dimensions (one for each layer) with `layers...`, set `num + 1` activation functions.

`-e` run tests

`-t batch_size rate epoch`  trains the network

`-s` run tests. Save as file in "params/" with the `name` being the success rate. If the network was loaded as a file the name will be concatenated with the original file name's postfix.

`-S name` run tests. Save as a file in "params/" with the `name`.

`-r` returns

## example I
### input
```sh
./NeuralNetworks 
-p 2
60 30
TANH TANH SIGMOID
-t 10 3 5
-e
-t 40 2 4
-e
-s
```

### output

```sh
-p 2
60 30
TANH TANH SIGMOID
-t 10 3 5
running  0  out of  5
running  1  out of  5
running  2  out of  5
running  3  out of  5
running  4  out of  5
done
-e
accuracy  0.8323
-t 40 2 4
running  0  out of  4
running  1  out of  4
running  2  out of  4
running  3  out of  4
done
-e
accuracy  0.8525
-s
error  0.8525
saved  0.8525
```

## example II
### input

```sh
./NeuralNetworks 
-lp 0.9336
-t 100 0.5 1
-e
-s
```

### output

```sh
-lp 0.9336
TANH
 60 784
RELU
 40  60
SIGMOID
 10  40
-t 100 0.5 1
running  0  out of  1
done
-e
accuracy  0.9358
-s
error  0.9358
saved  0.9358
```
