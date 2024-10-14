#include <iostream>
#include <nn.hpp>
#include <omp.h>
#include <random>
#include <stddef.h>
#include <stdexcept>
#include <vector>

using namespace std;

std::default_random_engine nn::e;
std::uniform_real_distribution<float> nn::dist(0.0f, 1.0f);

nn::Tensor::Tensor(vector<size_t> shape, size_t size)
    : shape(shape), data(vector<float>(size)), grad(vector<float>(size, 0.0f)), size(size) {
    size_t calculatedSize = 1;
    for (size_t dim : shape) {
        calculatedSize *= dim;
    }

    if (calculatedSize != size) {
        throw std::invalid_argument(
            "The product of dimensions in shape does not match the provided size");
    }

    for (size_t i = 0; i < size; i++) {
        data[i] = dist(e);
    }
};

void nn::Module::zero_grad() {
    for (auto &param : parameters()) {
        for (auto &el : param.grad) {
            el = 0.0f;
        }
    }
}

vector<nn::Tensor> nn::Module::parameters() {
    return vector<nn::Tensor>();
}

nn::FeedForwardNN::FeedForwardNN(size_t input_dim, size_t hidden_dim, size_t output_dim)
    : w1({input_dim, hidden_dim}, input_dim * hidden_dim),
      v({input_dim, hidden_dim}, input_dim * hidden_dim),
      w2({hidden_dim, output_dim}, input_dim * hidden_dim),
      b2({1, output_dim}, input_dim * hidden_dim) {}

vector<nn::Tensor> nn::FeedForwardNN::parameters() {
    return {w1, v, w2, b2};
}

nn::AdamW::AdamW(float lr, float beta_1, float beta_2, float eps, float weight_decay,
                 std::vector<nn::Tensor> parameters)
    : lr(lr), beta_1(beta_1), beta_2(beta_2), eps(eps), weight_decay(weight_decay) {
    for (auto &param : parameters) {
        m.push_back(std::vector<float>(param.size, 0.0f));
        v.push_back(std::vector<float>(param.size, 0.0f));
    }
};

void nn::AdamW::update(std::vector<nn::Tensor> parameters, int t) {
#pragma omp parallel for
    for (size_t i = 0; i < parameters.size(); ++i) {
        nn::Tensor &param = parameters[i];

#pragma omp parallel for
        for (size_t j = 0; j < param.size; ++j) {
            m[i][j] = beta_1 * m[i][j] + (1 - beta_1) * param.grad[j];
            v[i][j] = beta_2 * v[i][j] + (1 - beta_2) * (param.grad[j] * param.grad[j]);

            float m_hat = m[i][j] / (1 - std::pow(beta_1, t));
            float v_hat = v[i][j] / (1 - std::pow(beta_2, t));

            param.data[j] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param.data[j]);
        }
    }
}