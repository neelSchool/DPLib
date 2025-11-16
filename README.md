File upload for implemeters of such systems. 

Data source for mnist should be put in a folder labelled data in Algorithm1 folder, 

C++ Libtorch and OpenSSL are the only two external libraries used in this code, so please download those. 

Libtorch version is CPU only not CUDA or Rocm but code can be ran quicker through parallelization if a GPU is used.

Also currently only MNIST which is a simple CNN has had DP_SGD applied however this process can be applied to VGG 11 models like CIFAR 10 and 100 very easily by simply loading those models using libtorch instead of MNIST along with their respective datasets. 

Metrics to test for may include; Accuracy, Train Time (Batch and Overall) and Privacy Budget. 

Once Proving system is added all metrics to do with ZKPs can also be benchmarked. 

download the data files from https://yann.lecun.org/exdb/mnist/index.html
For Algorithm1 tests using CMake file:
```
mkdir build && cd build
cmake ..
make train -j4
```

For algorithm2 test as provided in any Code Editor or IDE.
For (Proving+Verifiying) GKR, Sumcheck and IVC reference this codebase https://github.com/zkPoTs/kaizen/tree/main

If you have any questions or need help understanding or improving upon any elements of the code, feel free to contact neel.mehendale@gmail.com
