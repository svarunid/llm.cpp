cmake_minimum_required(VERSION 3.30)
project(llm_cpp VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(train_gpt2 src/train_gpt2.cpp)
find_package(Arrow CONFIG REQUIRED)
target_link_libraries(train_gpt2 PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")

find_package(Parquet CONFIG REQUIRED)
target_link_libraries(train_gpt2 PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")

enable_testing()
add_subdirectory(tests)