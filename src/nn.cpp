#include <stdexcept>
#include <iostream>
#include <random>
#include <vector>

#include <stddef.h>

#include <nn.hpp>

using namespace std;

std::default_random_engine nn::e;
std::uniform_real_distribution<float> nn::dist(0.0f, 1.0f);

nn::Tensor::Tensor(vector<size_t> shape, size_t size) :
    shape(shape), data(vector<float>(size)),
    grad(vector<float>(size, 0.0f)), size(size) {
    size_t calculatedSize = 1;
    for (size_t dim : shape) {
        calculatedSize *= dim;
    }

    if (calculatedSize != size) {
        throw std::invalid_argument("The product of dimensions in shape does not match the provided size");
    }

    for (size_t i = 0; i < size; i++) {
        data[i] = dist(e);
    }
};

void nn::Module::zero_grad() {
    for (auto& param : parameters()) {
        for (auto& el : param.grad) {
            el = 0.0f;
        }
    }
}

vector<nn::Tensor> nn::Module::parameters() { return vector<nn::Tensor>(); }

nn::FeedForwardNN::FeedForwardNN(size_t input_dim, size_t hidden_dim, size_t output_dim) :
    w1({ input_dim, hidden_dim }, input_dim* hidden_dim),
    v({ input_dim, hidden_dim }, input_dim* hidden_dim),
    w2({ hidden_dim, output_dim }, input_dim* hidden_dim),
    b2({ 1, output_dim }, input_dim* hidden_dim) {}

vector<nn::Tensor> nn::FeedForwardNN::parameters() { return { w1, v, w2, b2 }; }

nn::AdamW::AdamW(
    float lr, float beta_1,
    float beta_2, float eps,
    float weight_decay, std::vector<Tensor> parameters
) :
    lr(lr), beta_1(beta_1), beta_2(beta_2),
    eps(eps), weight_decay(weight_decay) {
    std::vector<std::vector<float>> m, v;
    for (auto param : parameters) {
        m.push_back(std::vector<float>(param.size, 0.0f));
        v.push_back(std::vector<float>(param.size, 0.0f));
    }
};