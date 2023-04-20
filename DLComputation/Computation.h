#ifndef COMPUTATION_H_
#define COMPUTATION_H_

#include <iostream>
#include <torch/torch.h>

class mlp_computation : public torch::nn::Module
{
public:
    mlp_computation(int input_size, int hiden_size1, int hiden_size2, int output_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hiden_size1))),
          fc2(register_module("fc2", torch::nn::Linear(hiden_size1, hiden_size2))),
          fc3(register_module("fc3", torch::nn::Linear(hiden_size2, output_size))) {}
    ~mlp_computation() = default;
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::dropout(x, 0.2, is_training());
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

class FixedMLP : public torch::nn::Module
{
public:
    FixedMLP(int input_size, int hiden_size1, int hiden_size2, int output_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hiden_size1))),
          fc2(register_module("fc2", torch::nn::Linear(hiden_size1, hiden_size2))),
          fc3(register_module("fc3", torch::nn::Linear(hiden_size2, output_size)))
    {
        weight = torch::randn({hiden_size2, hiden_size2});
    }
    ~FixedMLP() = default;
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(x.mm(weight) + 1);
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        while (x.abs().sum().item<float>() > 1)
            x /= 2;
        return x;
    }

private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::Tensor weight;
};

class NestMLP : public torch::nn::Module
{
public:
    NestMLP() = default;
    ~NestMLP() = default;
    torch::Tensor forward(torch::Tensor x)
    {
        x = net->forward(x);
        return x;
    }

private:
    torch::nn::Sequential net{
        torch::nn::Linear(784, 256),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(256, 256),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.2),
        torch::nn::Linear(256, 10),
        torch::nn::LogSoftmax(1)};
};

class block1 : public torch::nn::Module
{
public:
    block1()
    {
        net = torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU(), torch::nn::Linear(8, 4), torch::nn::ReLU());
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = net->forward(x);
        return x;
    }

private:
    torch::nn::Sequential net{nullptr};
};

class block2 : public torch::nn::Module
{
public:
    block2()
    {
        net = torch::nn::Sequential(block1(), block1(), block1(), block1());
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = net->forward(x);
        return x;
    }

private:
    torch::nn::Sequential net{nullptr};
};

class MyLinear : public torch::nn::Module
{
public:
    MyLinear(int input_size, int output_size)
    {
        weight = register_parameter("weight", torch::randn({output_size, input_size}));
        bias = register_parameter("bias", torch::randn({output_size}));
    }
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.mm(weight.t()) + bias;
        x = torch::relu(x);
        return x;
    }

private:
    torch::Tensor weight;
    torch::Tensor bias;
};

#endif /* COMPUTATION_H_ */