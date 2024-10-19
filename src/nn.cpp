#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <nn.hpp>
#include <random>
#include <stddef.h>
#include <stdexcept>
#include <vector>

using namespace std;

/*Tensor with the given shape and size and populate elements using generator method*/
nn::Tensor::Tensor(vector<size_t> shape, size_t size, const std::function<float()> &gen)
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
        data[i] = gen();
        grad[i] = 0.0f;
    }
};

/*Tensor with the given shape and size and intialize elements to `init`*/
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

    for (size_t i = 0; i < size; ++i) {
        data[i] = init;
        grad[i] = 0.0f;
    }
};

/*Get module parameters.*/
std::vector<nn::Tensor *> nn::Module::parameters() {
    return std::vector<nn::Tensor *>();
}

/*Get module parameters along with activation*/
std::vector<nn::Tensor *> nn::Module::_parameters() {
    return std::vector<nn::Tensor *>();
}

/*Make tensor gradients zero.*/
void nn::Module::zero_grad() {
    for (auto &param : _parameters()) {
        for (auto &el : param->grad) {
            el = 0.0f;
        }
    }
}

/*Initialize FeedForwardNN with given vocabulary size and model dimensions.*/
nn::Embedding::Embedding(size_t vocab_size, size_t emb_dim, const std::function<float()> &gen)
    : emb(nn::Tensor({vocab_size, emb_dim}, vocab_size * emb_dim, gen)) {}

/*Get parameters of Embedding.*/
vector<nn::Tensor *> nn::Embedding::parameters() {
    return {&emb};
}

/*Get parameters of Embedding along with activation.*/
vector<nn::Tensor *> nn::Embedding::_parameters() {
    return {&emb};
}

/*Calculate the forward of Embedding.*/
nn::Tensor nn::Embedding::operator()(int token) {
    Tensor out({emb.shape.back()}, emb.shape.back(), 0.0f);
    for (size_t i = 0; i < emb.shape.back(); ++i) {
        out.data[i] = emb.data[i * token];
    }

    return out;
}

/*Backpropagation to find gradients of the embeddings.*/
void nn::Embedding::backward(int token, Tensor &dout) {
    for (size_t i = 0; i < emb.shape.back(); ++i) {
        emb.grad[token * emb.shape.back() + i] += dout.grad[i];
    }
}

/*Layer Normalization. Takes a 1D Tensor and noramlizes it.*/
nn::LayerNorm::LayerNorm(size_t input_dim, const std::function<float()> &gen)
    : w(nn::Tensor({input_dim}, input_dim, gen)), b(nn::Tensor({input_dim}, input_dim, gen)),
      mean(nn::Tensor({input_dim}, input_dim, 0.0f)),
      rstd(nn::Tensor({input_dim}, input_dim, 0.0f)) {}

/*Get parameters of LayerNorm.*/
vector<nn::Tensor *> nn::LayerNorm::parameters() {
    return {&w, &b};
}

/*Get parameters of LayerNorm along with activation.*/
vector<nn::Tensor *> nn::LayerNorm::_parameters() {
    return {&w, &b, &mean, &rstd};
}

/*Calculate forward pass for LayerNorm*/
nn::Tensor nn::LayerNorm::operator()(Tensor &x) {
    size_t rows = x.shape[0], cols = x.shape[1];

    // Calculate mean
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mean.data[i] += x.data[i + cols * j];
        }
        mean.data[i] /= cols;
    }

    // Calculate standard deviation
    for (size_t i = 0; i < rows; ++i) {
        float v = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float xshift = x.data[i + cols * j] - mean.data[i];
            v += xshift * xshift;
        }
        rstd.data[i] = 1.0f / std::sqrt((v / cols) + 1e-5f);
    }

    // Normalize, scale & shift inputs
    nn::Tensor out({rows, cols}, rows * cols, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i + cols * j;

            float n = rstd.data[i] * (x.data[index] - mean.data[i]);
            out.data[index] = n * w.data[i] + b.data[i];
        }
    }

    return out;
}

/*Backpropagation of LayerNorm to find gradients of the parameters.*/
nn::Tensor *nn::LayerNorm::backward(Tensor &x, Tensor &dout) {
    size_t rows = x.shape[0], cols = x.shape[1];

    std::vector<float> dnorm_mean(rows, 0.0f);
    std::vector<float> dnorm_norm_mean(rows, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i + cols * j;

            float dnorm = dout.grad[index] * w.data[i];
            dnorm_mean[i] += dnorm;
            dnorm_norm_mean[i] += dnorm * rstd.data[i] * (x.data[index] - mean.data[i]);
        }
        dnorm_mean[i] /= cols;
        dnorm_norm_mean[i] /= cols;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i + cols * j;

            float norm = rstd.data[i] * (x.data[index] - mean.data[i]);
            float dnorm = dout.grad[index] * w.data[i];

            b.grad[i] += dout.grad[index];
            w.grad[i] += norm * dout.grad[index];

            x.grad[index] += (dnorm - dnorm_mean[i] - norm * dnorm_norm_mean[i]) * rstd.data[i];
        }
    }

    return &x;
}

/*FeedForwardNN with given dimentions. Uses `SwiGLU` for activation.*/
nn::FeedForwardNN::FeedForwardNN(size_t input_dim, size_t hidden_dim, size_t output_dim,
                                 const std::function<float()> &gen)
    : w1({input_dim, hidden_dim}, input_dim * hidden_dim, gen),
      v({input_dim, hidden_dim}, input_dim * hidden_dim, gen),
      w2({hidden_dim, output_dim}, hidden_dim * output_dim, gen), b2({output_dim}, output_dim, gen),
      z1({hidden_dim}, hidden_dim, 0.0f), z2({hidden_dim}, hidden_dim, 0.0f),
      h({hidden_dim}, hidden_dim, 0.0f) {}

/*Get parameters of FeedForwardNN.*/
vector<nn::Tensor *> nn::FeedForwardNN::parameters() {
    return {&w1, &v, &w2, &b2};
}

/*Get parameters of FeedForwardNN along with activation.*/
vector<nn::Tensor *> nn::FeedForwardNN::_parameters() {
    return {&w1, &v, &w2, &b2, &z1, &z2, &h};
}

/*Calculate the forward of FeedForwardNN.*/
nn::Tensor nn::FeedForwardNN::operator()(nn::Tensor &x) {
    size_t hidden_dim = w1.shape.back();
    size_t output_dim = w2.shape.back();

    // w1 * x
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < x.size; ++j) {
            z1.data[i] += w1.data[i + hidden_dim * j] * x.data[j];
        }
        // Swish(z1)
        z1.data[i] = z1.data[i] / (1 + std::exp(-z1.data[i]));
    }

    // Swish(z1) * (v * x)
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < x.size; ++j) {
            z2.data[i] += v.data[i + hidden_dim * j] * x.data[j];
        }
        h.data[i] = z1.data[i] * z2.data[i];
    }

    // h * w2 + b2
    nn::Tensor y({output_dim}, output_dim, 0.0f);
    for (size_t i = 0; i < output_dim; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            y.data[i] += w2.data[i + hidden_dim * j] * h.data[j];
        }
        y.data[i] += b2.data[i];
    }

    return y;
}

/*Backpropagation of FeedForwardNN to find gradients of the parameters.*/
nn::Tensor *nn::FeedForwardNN::backward(nn::Tensor &x, nn::Tensor &dout) {
    size_t hidden_dim = w1.shape.back();
    size_t output_dim = w2.shape.back();

    // Gradient of h, w2 & b2.
    for (size_t i = 0; i < output_dim; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            w2.grad[i + output_dim * j] += dout.grad[i] * h.data[j];
            h.grad[j] += dout.grad[i] * w2.data[i + output_dim * j];
        }
        b2.grad[i] += dout.grad[i];
    }

    // Gradient of z1, z2 & v.
    for (size_t i = 0; i < hidden_dim; ++i) {
        z2.grad[i] = h.grad[i] * z1.data[i];

        for (size_t j = 0; j < x.size; ++j) {
            v.grad[i + hidden_dim * j] += z2.grad[i] * x.data[j];
        }

        float dswish = h.grad[i] * z2.data[i], exp_z1 = std::exp(z1.data[i]);
        z1.grad[i] = dswish * ((exp_z1 * (z1.data[i] + exp_z1 + 1)) / std::pow((exp_z1 + 1), 2));
    }

    // Gradient of w1 & x.
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < x.size; ++j) {
            w1.grad[i + hidden_dim * j] += z1.grad[i] * x.data[j];
            x.grad[j] += z1.grad[i] * w1.data[i + hidden_dim * j];
        }
    }

    return &x;
}

/*AdamW optimizer with weight decay.*/
nn::AdamW::AdamW(float lr, float beta_1, float beta_2, float eps, float weight_decay,
                 std::vector<nn::Tensor *> parameters)
    : lr(lr), beta_1(beta_1), beta_2(beta_2), eps(eps), weight_decay(weight_decay) {
    for (auto &param : parameters) {
        m.push_back(std::vector<float>(param->size, 0.0f));
        v.push_back(std::vector<float>(param->size, 0.0f));
    }
};

/*Update parameters based on AdamW*/
void nn::AdamW::update(std::vector<nn::Tensor *> parameters, int t) {
    for (size_t i = 0; i < parameters.size(); ++i) {
        nn::Tensor *param = parameters[i];
        for (size_t j = 0; j < param->size; ++j) {
            m[i][j] = beta_1 * m[i][j] + (1 - beta_1) * param->grad[j];
            v[i][j] = beta_2 * v[i][j] + (1 - beta_2) * (param->grad[j] * param->grad[j]);

            float m_hat = m[i][j] / (1 - std::pow(beta_1, t));
            float v_hat = v[i][j] / (1 - std::pow(beta_2, t));

            param->data[j] -=
                lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param->data[j]);
        }
    }
}