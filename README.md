# llm.cpp
Building LLM architectures from scratch using C++ (Inspired by [llm.c](https://github.com/karpathy/llm.c))

## Overview
I'm new to low-level languages like C/C++. Being a software engineer working on deep learning, I wanted to learn them to be able to better develop softwares that are efficient & performant. 

The goal of this project is, for me to able to write softwares comfortably in C/C++, learn more about system memory and also learn to program using CUDA APIs. 

Once I have accomplished above, I'll try to train a small version (124M) of GPT-2 on the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset by HuggingFace. Though I don't intend this model to compete with LLMs, I want the experience of orchestrating a workflow of training such large models.

## Structure
The code for different layer of a neural network lives in [`include/nn.hpp`](include/nn.hpp) file while the implementation can be found in [`src/nn.cpp`](src/nn.cpp). 

The library is tested using `gtest` framework and the test live in `tests/` directory.