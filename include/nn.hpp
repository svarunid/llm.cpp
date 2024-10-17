#pragma once

#include <random>
#include <stddef.h>
#include <vector>

namespace nn {
extern std::default_random_engine e;
extern std::uniform_real_distribution<float> dist;

struct Tensor {
    size_t size;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;

    Tensor(std::vector<size_t> shape, size_t size);
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
    Embedding(size_t vocab_size, size_t emb_dim);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(int token);
    void backward(int token, Tensor &dout);

  private:
    Tensor emb;
};

class LayerNorm : public Module {
  public:
    LayerNorm(size_t *input_dim);

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
    FeedForwardNN(size_t input_dim, size_t hidden_dim, size_t output_dim);

    std::vector<Tensor *> parameters() override;
    std::vector<Tensor *> _parameters() override;

    Tensor operator()(Tensor &x);
    Tensor *backward(Tensor &x, Tensor &dout);

  private:
    Tensor w1, v, w2, b2;
    Tensor z1, z2, h;
};
}; // namespace nn