#pragma once
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <random>

// ================================================================
//  BOX–MULLER GAUSSIAN NOISE
// ================================================================
inline double box_muller(double mean, double stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double u1 = dis(gen);
    double u2 = dis(gen);

    double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * M_PI * u2);
    return mean + stddev * z;
}

// Generate noise-like tensor
inline torch::Tensor gaussian_noise_like(const torch::Tensor& ref, double stddev) {
    auto noise = torch::zeros_like(ref);
    auto acc = noise.flatten(); // 1D view for iteration

    for (int64_t i = 0; i < acc.size(0); i++)
        acc[i] = box_muller(0.0, stddev);

    return noise;
}


// ================================================================
//  PER-SAMPLE GRADIENTS (single sample)
//  This uses: model->zero_grad() and model->forward()
// ================================================================
template <typename Model>
inline std::vector<torch::Tensor> compute_per_sample_gradients(
        Model& model,
        const torch::Tensor& x,
        const torch::Tensor& y)
{
    model->zero_grad();

    auto output = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(output, y);
    loss.backward();

    std::vector<torch::Tensor> grads;
    grads.reserve(model->parameters().size());

    for (auto& p : model->parameters()) {
        grads.push_back(p.grad().detach().clone());
    }

    return grads;
}


// ================================================================
//  DP-SGD UPDATE STEP (Algorithm 1)
// ================================================================
template <typename Model>
inline void dp_sgd_step(
    Model& model,
    const torch::Tensor& batch_x,
    const torch::Tensor& batch_y,
    double C,       // clipping norm
    double eta,     // learning rate
    double sigma)   // noise multiplier
{
    int batch_size = batch_x.size(0);

    // Store: per-sample gradient vectors
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);


    // =================================================================
    // 1. PER-SAMPLE GRADIENTS
    // =================================================================
    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);   // [1, 1, 28, 28]
        auto y_i = batch_y[i].unsqueeze(0);

        auto grads = compute_per_sample_gradients(model, x_i, y_i);
        per_sample_grads.push_back(grads);
    }


    // =================================================================
    // 2. CLIP GRADIENTS
    // =================================================================
    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;

        // Compute squared norm
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();

        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);

        // Apply clipping
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }


    // =================================================================
    // 3. AVERAGE CLIPPED GRADIENT
    // =================================================================
    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());

    // Initialize zeros
    for (auto& p : model->parameters()) {
        avg_grads.push_back(torch::zeros_like(p));
    }

    // Sum
    for (int i = 0; i < batch_size; i++) {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            avg_grads[k] += per_sample_grads[i][k];
        }
    }

    // Divide by batch size
    for (auto& g : avg_grads)
        g /= batch_size;


    // =================================================================
    // 4. ADD GAUSSIAN NOISE
    // =================================================================
    double noise_std = sigma * C;

    for (size_t k = 0; k < avg_grads.size(); k++) {
        auto noise = gaussian_noise_like(avg_grads[k], noise_std);
        avg_grads[k] += noise;
    }


    // =================================================================
    // 5. WEIGHT UPDATE
    // =================================================================
    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * avg_grads[k]);
        k++;
    }
}
