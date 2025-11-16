File upload for implemeters of such systems. 

Data source for mnist should be put in a folder labelled data in Algorithm1 folder, 

C++ Libtorch and OpenSSL are the only two external libraries used in this code, so please download those. 
Libtorch version is CPU onlh not CUDA or Rocm but code can be ran quicker through parallelization if a GPU is used.

download the data files from https://yann.lecun.org/exdb/mnist/index.html
For Algorithm1 tests using CMake file:
```
mkdir build && cd build
cmake ..
make train -j4
```

For algorithm2 test as provided in any Code Editor or IDE.
For (Proving+Verifiying) GKR, Sumcheck and IVC reference this codebase https://github.com/zkPoTs/kaizen/tree/main
