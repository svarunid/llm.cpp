#include <gtest/gtest.h>
#include <nn.hpp>
#include <stdexcept>

using namespace nn;

TEST(Tensor, ShapeMismatch) {
    std::vector<size_t> shape{10, 5};
    EXPECT_THROW(Tensor x(shape, 60), std::invalid_argument);
}

TEST(Tensor, HasCorrectSize) {
    std::vector<size_t> shape{10, 5};
    Tensor x(shape, 50);
    EXPECT_EQ(x.data.size(), 50);
    EXPECT_EQ(x.grad.size(), 50);
}