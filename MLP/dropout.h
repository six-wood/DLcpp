#ifndef DROPOUT_H
#define DROPOUT_H

#include <torch/torch.h>

torch::Tensor dropout_layer(torch::Tensor x, float dropout);

class mlp_dropout : public torch::nn::Module
{
public:
    mlp_dropout(int input_size, int hidden_size1, int hidden_size2, int output_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size1))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size1, hidden_size2))),
          fc3(register_module("fc3", torch::nn::Linear(hidden_size2, output_size))) {}
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

#endif /* DROPOUT_H */