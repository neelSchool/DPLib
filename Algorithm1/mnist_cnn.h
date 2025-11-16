#pragma once
#include <torch/torch.h>

// ===============================
// Simple CNN for MNIST (28x28x1)
// ===============================
struct MNISTCNNImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    MNISTCNNImpl() {
        // 1 input channel (grayscale), 32 output channels, 3x3 kernel
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)
        ));

        // 32 → 64 channels
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)
        ));

        // After 2 convolutions and 2 max-pools:
        // Input: 28x28
        // After pool → 14x14
        // After pool → 7x7
        // 64 channels → 64 * 7 * 7 = 3136

        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);

        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);

        x = x.view({x.size(0), -1});  // flatten

        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);  // logits (no softmax for cross_entropy)

        return x;
    }
};

TORCH_MODULE(MNISTCNN);
