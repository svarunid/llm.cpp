#pragma once

#include <functional>
#include <random>
#include <stddef.h>
#include <vector>

namespace nn {
struct Tensor {
    size_t size;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;

    Tensor(std::vector<size_t> shape, size_t size, const std::function<float()> &gen);
    Tensor(std::vector<size_t> shape, size_t size, float init);
};

struct Module {
    virtual std::vector<Tensor *> parameters();
    virtual std::vector<Tensor *> _parameters();
    void zero_grad();
};

class AdamW {
  public:
    AdamW(float lr, float beta_1, float beta_2, float eps, float weight_decay,
          std::vector<Tensor *> parameters);

    void update(std::vector<Tensor *> parameters, int t);

  private:
    float lr, beta_1, beta_2, eps, weight_decay;
    std::vector<std::vector<float>> m, v;
};

class Embedding : public Module {
  public:
    Embedding(size_t vocab_size, size_t emb_dim, const std::function<float()> &gen);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(std::vector<int> tokens);
    void backward(std::vector<int> tokens, Tensor &dout);

  private:
    Tensor emb;
};

class LayerNorm : public Module {
  public:
    LayerNorm(size_t input_dim, const std::function<float()> &gen);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(Tensor &x);
    Tensor *backward(Tensor &x, Tensor &dout);

  private:
    Tensor w, b;
    Tensor mean, rstd;
};

class FeedForwardNN : public Module {
  public:
    FeedForwardNN(size_t num_batch, size_t input_dim, size_t hidden_dim, size_t output_dim,
                  const std::function<float()> &gen);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(Tensor &x);
    Tensor *backward(Tensor &x, Tensor &dout);

  private:
    Tensor w1, v, w2, b2;
    Tensor z1, z2, h;
};

class MultiHeadAttention : public Module {
  public:
    MultiHeadAttention(size_t emb_dim, size_t num_heads, const std::function<float()> &gen);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(Tensor &x);
    Tensor *backward(Tensor &x, Tensor &dout);

  private:
    size_t num_heads;
    Tensor wq, wk, wv;
};

Tensor softmax(Tensor &x);
Tensor softmax_backward(Tensor &x, Tensor &dout);

Tensor cross_entropy_loss(Tensor &x, Tensor &y);
Tensor cross_entropy_loss_backward(Tensor &x, Tensor &y, Tensor &dout);
}; // namespace nn