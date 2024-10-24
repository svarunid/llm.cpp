#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <nn.hpp>
#include <stddef.h>
#include <stdexcept>
#include <vector>

/*Tensor with the given shape and size and populate elements using generator method*/
nn::Tensor::Tensor(std::vector<size_t> shape, size_t size, const std::function<float()> &gen)
    : shape(shape), data(std::vector<float>(size)), grad(std::vector<float>(size, 0.0f)),
      size(size) {
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
nn::Tensor::Tensor(std::vector<size_t> shape, size_t size, float init)
    : shape(shape), data(std::vector<float>(size)), grad(std::vector<float>(size, 0.0f)),
      size(size) {
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

nn::Tensor::Tensor() : shape({}), size(0), data({}), grad({}) {}

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

/*Initialize FeedForwardNN with given vocabulary size and model dimensions.*/
nn::Embedding::Embedding(size_t vocab_size, size_t emb_dim, const std::function<float()> &gen)
    : emb(nn::Tensor({vocab_size, emb_dim}, vocab_size * emb_dim, gen)) {}

/*Get parameters of Embedding.*/
std::vector<nn::Tensor *> nn::Embedding::parameters() {
    return {&emb};
}

/*Get parameters of Embedding along with activation.*/
std::vector<nn::Tensor *> nn::Embedding::_parameters() {
    return {&emb};
}

/*Calculate the forward of Embedding.*/
nn::Tensor nn::Embedding::operator()(std::vector<int> tokens) {
    Tensor out({tokens.size(), emb.shape.back()}, tokens.size() * emb.shape.back(), 0.0f);
    for (size_t i = 0; i < tokens.size(); ++i) {
        for (size_t j = 0; j < emb.shape.back(); ++j) {
            out.data[emb.shape.back() * i + j] = emb.data[tokens[i] * emb.shape.back() + j];
        }
    }

    return out;
}

/*Backpropagation to find gradients of the embeddings.*/
void nn::Embedding::backward(std::vector<int> tokens, Tensor &out) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        for (size_t j = 0; j < emb.shape.back(); ++j) {
            emb.grad[tokens[i] * emb.shape.back() + j] += out.grad[emb.shape.back() * i + j];
        }
    }
}

/*Layer Normalization. Takes a 1D Tensor and noramlizes it.*/
nn::LayerNorm::LayerNorm(size_t input_dim, const std::function<float()> &gen)
    : w(nn::Tensor({input_dim}, input_dim, gen)), b(nn::Tensor({input_dim}, input_dim, gen)),
      mean(nn::Tensor({input_dim}, input_dim, 0.0f)),
      rstd(nn::Tensor({input_dim}, input_dim, 0.0f)) {}

/*Get parameters of LayerNorm.*/
std::vector<nn::Tensor *> nn::LayerNorm::parameters() {
    return {&w, &b};
}

/*Get parameters of LayerNorm along with activation.*/
std::vector<nn::Tensor *> nn::LayerNorm::_parameters() {
    return {&w, &b, &mean, &rstd};
}

/*Calculate forward pass for LayerNorm*/
nn::Tensor nn::LayerNorm::operator()(Tensor &x) {
    size_t rows = x.shape[0], cols = x.shape[1];

    // Calculate mean
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mean.data[i] += x.data[i * rows + j];
        }
        mean.data[i] /= cols;
    }

    // Calculate standard deviation
    for (size_t i = 0; i < rows; ++i) {
        float v = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float xshift = x.data[i * rows + j] - mean.data[i];
            v += xshift * xshift;
        }
        rstd.data[i] = 1.0f / std::sqrt((v / cols) + 1e-5f);
    }

    // Normalize, scale & shift inputs
    nn::Tensor out({rows, cols}, rows * cols, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * rows + j;

            float n = rstd.data[i] * (x.data[index] - mean.data[i]);
            out.data[index] = n * w.data[i] + b.data[i];
        }
    }

    return out;
}

/*Backpropagation of LayerNorm to find gradients of the parameters.*/
nn::Tensor *nn::LayerNorm::backward(Tensor &x, Tensor &out) {
    size_t rows = x.shape[0], cols = x.shape[1];

    std::vector<float> dnorm_mean(rows, 0.0f);
    std::vector<float> dnorm_norm_mean(rows, 0.0f);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * rows + j;

            float dnorm = out.grad[index] * w.data[i];
            dnorm_mean[i] += dnorm;
            dnorm_norm_mean[i] += dnorm * rstd.data[i] * (x.data[index] - mean.data[i]);
        }
        dnorm_mean[i] /= cols;
        dnorm_norm_mean[i] /= cols;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * rows + j;

            float norm = rstd.data[i] * (x.data[index] - mean.data[i]);
            float dnorm = out.grad[index] * w.data[i];

            b.grad[i] += out.grad[index];
            w.grad[i] += norm * out.grad[index];

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
      w2({hidden_dim, output_dim}, hidden_dim * output_dim, gen),
      b2({output_dim}, output_dim, gen) {}

/*Get parameters of FeedForwardNN.*/
std::vector<nn::Tensor *> nn::FeedForwardNN::parameters() {
    return {&w1, &v, &w2, &b2};
}

/*Get parameters of FeedForwardNN along with activation.*/
std::vector<nn::Tensor *> nn::FeedForwardNN::_parameters() {
    return {&w1, &v, &w2, &b2, &z1, &z2, &h};
}

/*Calculate the forward of FeedForwardNN.*/
nn::Tensor nn::FeedForwardNN::operator()(nn::Tensor &x) {
    size_t input_dim = x.shape.back();
    size_t hidden_dim = w1.shape.back();
    size_t output_dim = w2.shape.back();

    // Initialize activation tensor if not initialized.
    if (z1.size == 0) {
        z1 = nn::Tensor({x.shape[0], hidden_dim}, x.shape[0] * hidden_dim, 0.0f);
        z2 = nn::Tensor({x.shape[0], hidden_dim}, x.shape[0] * hidden_dim, 0.0f);
        h = nn::Tensor({x.shape[0], hidden_dim}, x.shape[0] * hidden_dim, 0.0f);
    }

    // w1 * x
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = hidden_dim * i;
        for (size_t j = 0; j < hidden_dim; ++j) {
            const size_t embj = batchi + j;
            for (size_t k = 0; k < input_dim; ++k) {
                z1.data[embj] += w1.data[j + hidden_dim * k] * x.data[input_dim * i + k];
            }
            // Swish(z1)
            z1.data[embj] = z1.data[embj] / (1 + std::exp(-z1.data[embj]));
        }
    }

    // Swish(z1) * (v * x)
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = hidden_dim * i;
        for (size_t j = 0; j < hidden_dim; ++j) {
            const size_t embj = batchi + j;
            for (size_t k = 0; k < input_dim; ++k) {
                z2.data[embj] += v.data[j + hidden_dim * k] * x.data[input_dim * i + k];
            }
            h.data[embj] = z1.data[embj] * z2.data[embj];
        }
    }

    // h * w2 + b2
    nn::Tensor y({x.shape[0], output_dim}, output_dim, 0.0f);
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = output_dim * i;
        for (size_t j = 0; j < output_dim; ++j) {
            const size_t embj = batchi + j;
            for (size_t k = 0; k < hidden_dim; ++k) {
                y.data[embj] += w2.data[j + output_dim * k] * h.data[hidden_dim * i + k];
            }
            y.data[embj] += b2.data[j];
        }
    }

    return y;
}

/*Backpropagation of FeedForwardNN to find gradients of the parameters.*/
nn::Tensor *nn::FeedForwardNN::backward(nn::Tensor &x, nn::Tensor &out) {
    size_t input_dim = x.shape.back();
    size_t hidden_dim = w1.shape.back();
    size_t output_dim = w2.shape.back();

    // Gradient of h, w2 & b2.
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = output_dim * i;
        for (size_t j = 0; j < output_dim; ++j) {
            const size_t embj = batchi + j;
            for (size_t k = 0; k < hidden_dim; ++k) {
                w2.grad[j + output_dim * k] += out.grad[embj] * h.data[hidden_dim * i + k];
                h.grad[k] += out.grad[embj] * w2.data[j + output_dim * k];
            }
            b2.grad[j] += out.grad[embj];
        }
    }

    // Gradient of z1, z2 & v.
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = hidden_dim * i;
        for (size_t j = 0; j < hidden_dim; ++j) {
            const size_t embj = batchi + j;
            z2.grad[embj] = h.grad[embj] * z1.data[embj];

            for (size_t k = 0; k < input_dim; ++k) {
                v.grad[j + hidden_dim * k] += z2.grad[embj] * x.data[input_dim * i + k];
            }

            float dswish = h.grad[embj] * z2.data[embj], exp_z1 = std::exp(z1.data[embj]);
            z1.grad[embj] =
                dswish * ((exp_z1 * (z1.data[embj] + exp_z1 + 1)) / std::pow((exp_z1 + 1), 2));
        }
    }

    // Gradient of w1 & x.
    for (size_t i = 0; i < x.shape[0]; ++i) {
        const size_t batchi = hidden_dim * i;
        for (size_t j = 0; j < hidden_dim; ++j) {
            const size_t embj = batchi + j;
            for (size_t k = 0; k < input_dim; ++k) {
                w1.grad[j + hidden_dim * k] += z1.grad[embj] * x.data[input_dim * i + k];
                x.grad[embj] += z1.grad[embj] * w1.data[j + hidden_dim * k];
            }
        }
    }

    return &x;
}

/*MultiHeadAttention with dimension `emb_size`.*/
nn::MultiHeadAttention::MultiHeadAttention(size_t emb_dim, size_t num_heads,
                                           const std::function<float()> &gen)
    : num_heads(num_heads) {
    assert(emb_dim % num_heads == 0);
    wq = nn::Tensor({emb_dim, emb_dim}, emb_dim * emb_dim, gen);
    wk = nn::Tensor({emb_dim, emb_dim}, emb_dim * emb_dim, gen);
    wv = nn::Tensor({emb_dim, emb_dim}, emb_dim * emb_dim, gen);
    wo = nn::Tensor({emb_dim, emb_dim}, emb_dim * emb_dim, gen);
}

/*Get parameters of MultiHeadAttention.*/
std::vector<nn::Tensor *> nn::MultiHeadAttention::parameters() {
    return {&wq, &wk, &wv, &wo};
}

/*Get parameters of MultiHeadAttention with activations.*/
std::vector<nn::Tensor *> nn::MultiHeadAttention::_parameters() {
    return {&wq, &wk, &wv, &wo, &q, &k, &v, &qk};
}

nn::Tensor nn::MultiHeadAttention::operator()(nn::Tensor &x) {
    size_t seq = x.shape[0], emb = x.shape[1], dk = emb / num_heads;
    nn::Tensor out(x.shape, x.size, 0.0f);

    if (q.size == 0) {
        q = nn::Tensor({num_heads, seq, emb / num_heads}, x.size, 0.0f),
        k = nn::Tensor({num_heads, seq, emb / num_heads}, x.size, 0.0f),
        v = nn::Tensor({num_heads, seq, emb / num_heads}, x.size, 0.0f),
        qk = nn::Tensor({num_heads, seq, seq}, num_heads * seq * seq, 0.0f);
    }

    // Compute q, k, v by multiplying the input x with respective weight matrices
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t dki = 0; dki < dk; ++dki) {
                const size_t idx = h * seq * dk + s * dk + dki;
                for (size_t e = 0; e < emb; ++e) {
                    const size_t x_idx = s * emb + e;
                    const size_t w_idx = e * emb + h * dk + dki;
                    q.data[idx] += x.data[x_idx] * wq.data[w_idx];
                    k.data[idx] += x.data[x_idx] * wk.data[w_idx];
                    v.data[idx] += x.data[x_idx] * wv.data[w_idx];
                }
            }
        }
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(emb / num_heads));
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq; ++i) {
            for (size_t j = 0; j < seq; ++j) {
                for (size_t dki = 0; dki < dk; ++dki) {
                    qk.data[h * seq * seq + i * seq + j] +=
                        q.data[h * seq * dk + i * dk + dki] * k.data[h * seq * dk + j * dk + dki];
                }
                qk.data[h * seq * seq + i * seq + j] *= scale;
            }
        }
    }

    // Apply softmax to qk along the sequence dimension
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq; ++i) {
            // Calculate the sum of exponentials for normalization
            float exp_sum = 0.0f;
            for (size_t j = 0; j < seq; ++j) {
                qk.data[h * seq * seq + i * seq + j] =
                    std::exp(qk.data[h * seq * seq + i * seq + j]);
                exp_sum += qk.data[h * seq * seq + i * seq + j];
            }
            // Normalize each value in qk
            for (size_t j = 0; j < seq; ++j) {
                qk.data[h * seq * seq + i * seq + j] /= exp_sum;
            }
        }
    }

    return out;
}

nn::Tensor nn::softmax(nn::Tensor &x, int temp = 1) {
    nn::Tensor out(x.shape, x.size, 0.0f);

    size_t batch = x.shape[0], emb = x.shape[1];
    for (size_t i = 0; i < batch; ++i) {
        float max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < emb; ++j) {
            max = std::max(max, x.data[emb * i + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < emb; ++j) {
            out.data[emb * i + j] = std::exp((x.data[emb * i + j] - max) / temp);
            sum += out.data[emb * i + j];
        }

        for (size_t j = 0; j < emb; ++j) {
            out.data[emb * i + j] /= sum;
        }
    }

    return out;
}

nn::Tensor *nn::softmax_backward(nn::Tensor &x, nn::Tensor &out, int temp = 1) {
    size_t batch = x.shape[0], emb = x.shape[1];

    for (size_t i = 0; i < batch; ++i) {
        const size_t batchi = emb * i;
        float max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < emb; ++j) {
            max = std::max(max, x.data[batchi + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < emb; ++j) {
            sum += std::exp((x.data[batchi + j] - max) / temp);
        }

        float dsum = 0.0f;
        for (size_t j = 0; j < emb; ++j) {
            dsum += out.grad[batchi + j] * out.data[batchi + j];
        }

        for (size_t j = 0; j < emb; ++j) {
            x.grad[batchi + j] += out.data[batchi + j] / temp * (out.grad[batchi + j] * sum - dsum);
        }
    }

    return &x;
}