include_directories(../include)

find_package(GTest CONFIG REQUIRED)

add_executable(nn_test nn_test.cpp ../src/nn.cpp)

target_link_libraries(nn_test GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(nn_test)