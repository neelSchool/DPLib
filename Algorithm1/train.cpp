#include <torch/torch.h>
#include "dp_sgd_libtorch.h"
#include "mnist_cnn.h"

int main() {
    torch::manual_seed(0);

    MNISTCNN model;
    model->to(torch::kCPU);

    // MNIST dataset
    auto dataset = torch::data::datasets::MNIST("/home/neel/dpsgd_libtorch/data")
                   .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                   .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(32)
    );


    

    double C = 1.0;
    double eta = 0.01;
    double sigma = 1.1;

    // Training loop
    for (int epoch = 0; epoch < 1; epoch++) {
        int batch_idx = 0;
        for (auto& batch : *data_loader) {
            dp_sgd_step(model, batch.data, batch.target, C, eta, sigma);

            if (batch_idx % 100 == 0)
                std::cout << "Batch " << batch_idx << " done\n";

            batch_idx++;
        }
    }

    std::cout << "Training complete.\n";
    return 0;
}
