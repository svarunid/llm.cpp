#include <gtest/gtest.h>
#include <nn.hpp>
#include <random>
#include <stdexcept>

using namespace nn;

// Test if the tensor is created with appropriate shapes.
TEST(Tensor, ShapeMismatch) {
    std::default_random_engine e;
    std::uniform_real_distribution dist(0.0f, 1.0f);

    std::vector<size_t> shape{10, 5};
    const auto gen = [&e, &dist]() { return dist(e); };

    EXPECT_THROW(Tensor x(shape, 60, gen), std::invalid_argument)
        << "Doesn't throw errors when there is a shape & size mismatch";
}

// Test if the initialized tensor has correct size.
TEST(Tensor, HasCorrectSize) {
    std::default_random_engine e;
    std::uniform_real_distribution dist(0.0f, 1.0f);

    std::vector<size_t> shape{10, 5};
    const auto gen = [&e, &dist]() { return dist(e); };

    Tensor x(shape, 50, gen);
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

class FeedForwardNNTest : public testing::Test {
  protected:
    FeedForwardNNTest() : ffnn(3, 2, 1, init()) {}

    FeedForwardNN ffnn;
};

TEST_F(FeedForwardNNTest, CheckParameterList) {
    std::vector<Tensor *> model_parameters = ffnn.parameters();
    std::vector<Tensor *> model_parameters_with_activations = ffnn._parameters();

    ASSERT_EQ(model_parameters.size(), 4)
        << "Model Parameters: " << 4 << " expected " << model_parameters.size() << " received.";
    ASSERT_EQ(model_parameters_with_activations.size(), 7)
        << "Model parameters with activations: " << 7 << " expected "
        << model_parameters_with_activations.size() << " received.";
}

TEST_F(FeedForwardNNTest, InitCheck) {
    std::vector<Tensor *> parameters = ffnn.parameters();

    // Check if the parameters have correct shapes.
    ASSERT_EQ(parameters[0]->shape, (std::vector<size_t>{3, 2}));
    ASSERT_EQ(parameters[1]->shape, (std::vector<size_t>{3, 2}));
    ASSERT_EQ(parameters[2]->shape, (std::vector<size_t>{2, 1}));
    ASSERT_EQ(parameters[3]->shape, (std::vector<size_t>{1}));
}

TEST_F(FeedForwardNNTest, ForwardAndBackwardPass) {
    Tensor x({3}, 3, 3.0f);

    Tensor y = ffnn(x);

    y.grad[0] = 0.5f;
    ffnn.backward(x, y);

    EXPECT_FLOAT_EQ(28446, y.data[0]);
    EXPECT_FLOAT_EQ(1786.5, x.grad[0]);
    EXPECT_FLOAT_EQ(4099.5, x.grad[1]);
    EXPECT_FLOAT_EQ(6412.5, x.grad[2]);
}