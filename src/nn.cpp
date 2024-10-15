#include <algorithm>
#include <cassert>
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
        grad[i] = 0.0f;
    }
};

nn::Tensor::Tensor(vector<size_t> shape, size_t size, float init)
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
        data[i] = init;
        grad[i] = 0.0f;
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
      b2({1, output_dim}, input_dim * hidden_dim), z1({1, hidden_dim}, hidden_dim, 0.0f),
      z2({1, hidden_dim}, hidden_dim, 0.0f), h({1, hidden_dim}, hidden_dim, 0.0f) {}

vector<nn::Tensor> nn::FeedForwardNN::parameters() {
    return {w1, v, w2, b2, z1, z2, h};
}

nn::Tensor nn::FeedForwardNN::forward(nn::Tensor &x) {
    std::vector<size_t> expected_shape = {1, w1.shape[0]};
    assert(x.shape == expected_shape);

    size_t hidden_dim = w1.shape.back();
    size_t output_dim = w2.shape.back();

    // w1 * x
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < x.size; ++j) {
            z1.data[i] += w1.data[i * x.size + j] * x.data[j];
        }
        z1.data[i] = z1.data[i] / (1 + std::exp(-z1.data[i]));
    }

    // Swish(z1) * Linear(x)
    for (size_t i = 0; i < hidden_dim; ++i) {
        z2.data[i] = 0.0f;
        for (size_t j = 0; j < x.size; ++j) {
            z2.data[i] += v.data[i * x.size + j] * x.data[j];
        }
        h.data[i] = z1.data[i] * z2.data[i];
    }

    // h * w2 + b2
    nn::Tensor y({1, output_dim}, output_dim);
    for (size_t i = 0; i < output_dim; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            y.data[i] += w2.data[i * hidden_dim + j] * h.data[j];
        }
        y.data[i] += b2.data[i];
    }

    return y;
}

nn::Tensor nn::FeedForwardNN::backward(nn::Tensor &x, nn::Tensor &dout) {
    assert(dout.shape == b2.shape);

    size_t hidden_dim = w1.shape[0];
    std::vector<float> dh(hidden_dim, 0.0f);
    for (size_t i = 0; i < w2.shape.back(); ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            w2.grad[i * hidden_dim + j] += dout.grad[i] * h.data[j];
            dh[j] += dout.grad[i] * w2.data[i * hidden_dim + j];
        }
        b2.grad[i] += dout.grad[i];
    }

    std::vector<float> dz1(hidden_dim, 0.0f);
    for (size_t i = 0; i < hidden_dim; ++i) {
        z2.grad[i] = dh[i] * z1.data[i];

        for (size_t j = 0; j < x.size; ++j) {
            v.grad[i * x.size + j] += z2.grad[i] * x.data[j]; // Gradient of v
        }

        float dswish = dh[i] * z2.data[i], exp_z1 = std::exp(z1.data[i]);
        dz1[i] = dswish * ((exp_z1 * (z1.data[i] + exp_z1 + 1)) / std::pow((exp_z1 + 1), 2));
    }

    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < x.size; ++j) {
            w1.grad[i * x.size + j] += dz1[i] * x.data[j];
            x.grad[j] += dz1[i] * w1.data[i * x.size + j];
        }
    }

    return x;
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
        for (size_t j = 0; j < param.size; ++j) {
            m[i][j] = beta_1 * m[i][j] + (1 - beta_1) * param.grad[j];
            v[i][j] = beta_2 * v[i][j] + (1 - beta_2) * (param.grad[j] * param.grad[j]);

            float m_hat = m[i][j] / (1 - std::pow(beta_1, t));
            float v_hat = v[i][j] / (1 - std::pow(beta_2, t));

            param.data[j] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param.data[j]);
        }
    }
}