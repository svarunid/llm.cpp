#include <gtest/gtest.h>
#include <nn.hpp>
#include <random>
#include <stdexcept>

// Test if the tensor is created with appropriate shapes.
TEST(Tensor, ShapeMismatch) {
    std::default_random_engine e;
    std::uniform_real_distribution dist(0.0f, 1.0f);

    std::vector<size_t> shape{10, 5};
    const auto gen = [&e, &dist]() { return dist(e); };

    EXPECT_THROW(nn::Tensor x(shape, 60, gen), std::invalid_argument)
        << "Doesn't throw errors when there is a shape & size mismatch";
}

// Test if the initialized tensor has correct size.
TEST(Tensor, HasCorrectSize) {
    std::default_random_engine e;
    std::uniform_real_distribution dist(0.0f, 1.0f);

    std::vector<size_t> shape{10, 5};
    const auto gen = [&e, &dist]() { return dist(e); };

    nn::Tensor x(shape, 50, gen);
    EXPECT_EQ(x.data.size(), 50);
    EXPECT_EQ(x.grad.size(), 50);
}

// Generate a sequence of number starting from 0.
std::function<float()> init() {
    float counter = 0.0f;
    const auto gen = [counter]() mutable {
        counter = counter + 1.0f;
        return counter;
    };
    return gen;
}

// Testing layer normalization module
// Fixture for LayerNorm.
class LayerNormTest : public testing::Test {
  protected:
    LayerNormTest() : ln(3, init()) {}

    nn::LayerNorm ln;
};

TEST_F(LayerNormTest, CheckParameterList) {
    std::vector<nn::Tensor *> model_parameters = ln.parameters();
    std::vector<nn::Tensor *> model_parameters_with_activations = ln._parameters();

    ASSERT_EQ(model_parameters.size(), 2)
        << "Model Parameters: " << 2 << " expected " << model_parameters.size() << " received.";
    ASSERT_EQ(model_parameters_with_activations.size(), 4)
        << "Model parameters with activations: " << 4 << " expected "
        << model_parameters_with_activations.size() << " received.";
}

TEST_F(LayerNormTest, InitCheck) {
    std::vector<nn::Tensor *> parameters = ln.parameters();

    // Check if the parameters have correct shapes.
    ASSERT_EQ(parameters[0]->shape, (std::vector<size_t>{3}));
    ASSERT_EQ(parameters[1]->shape, (std::vector<size_t>{3}));
}

TEST_F(LayerNormTest, ForwardAndBackwardPass) {
    nn::Tensor x({1, 3}, 3, 3.0f);

    nn::Tensor y = ln(x);

    for (size_t i = 0; i < y.size; ++i) {
        y.grad[i] = 0.5f;
    }
    ln.backward(x, y);

    EXPECT_FLOAT_EQ(4, y.data[0]);
    EXPECT_FLOAT_EQ(4, y.data[1]);
    EXPECT_FLOAT_EQ(4, y.data[2]);

    EXPECT_FLOAT_EQ(0, x.grad[0]);
    EXPECT_FLOAT_EQ(0, x.grad[1]);
    EXPECT_FLOAT_EQ(0, x.grad[2]);
}

// Testing feed forword neural network module.
// Fixture for FeedForwardNN.
class FeedForwardNNTest : public testing::Test {
  protected:
    FeedForwardNNTest() : ffnn(1, 3, 2, 1, init()) {}

    nn::FeedForwardNN ffnn;
};

TEST_F(FeedForwardNNTest, CheckParameterList) {
    std::vector<nn::Tensor *> model_parameters = ffnn.parameters();
    std::vector<nn::Tensor *> model_parameters_with_activations = ffnn._parameters();

    ASSERT_EQ(model_parameters.size(), 4)
        << "Model Parameters: " << 4 << " expected " << model_parameters.size() << " received.";
    ASSERT_EQ(model_parameters_with_activations.size(), 7)
        << "Model parameters with activations: " << 7 << " expected "
        << model_parameters_with_activations.size() << " received.";
}

TEST_F(FeedForwardNNTest, InitCheck) {
    std::vector<nn::Tensor *> parameters = ffnn.parameters();

    // Check if the parameters have correct shapes.
    ASSERT_EQ(parameters[0]->shape, (std::vector<size_t>{3, 2}));
    ASSERT_EQ(parameters[1]->shape, (std::vector<size_t>{3, 2}));
    ASSERT_EQ(parameters[2]->shape, (std::vector<size_t>{2, 1}));
    ASSERT_EQ(parameters[3]->shape, (std::vector<size_t>{1}));
}

TEST_F(FeedForwardNNTest, ForwardAndBackwardPass) {
    nn::Tensor x({1, 3}, 3, 3.0f);

    nn::Tensor y = ffnn(x);

    y.grad[0] = 0.5f;
    ffnn.backward(x, y);

    EXPECT_FLOAT_EQ(73806, y.data[0]);

    EXPECT_FLOAT_EQ(4738.5, x.grad[0]);
    EXPECT_FLOAT_EQ(7560, x.grad[1]);
    EXPECT_FLOAT_EQ(0, x.grad[2]);
}