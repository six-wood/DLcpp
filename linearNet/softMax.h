#ifndef _SOFT_MAX_H_
#define _SOFT_MAX_H_

#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class softMax : public torch::nn::Module
{
public:
    softMax(int num_inputs, int num_outputs): linear(register_module("fc", torch::nn::Linear(num_inputs, num_outputs))) {}

    inline torch::Tensor forward(torch::Tensor x)
    {

        return torch::log_softmax(linear->forward(x.reshape({x.size(0), 784})), /*dim=*/1);
    }

private:
    torch::nn::Linear linear{nullptr};
};

class Accumlator
{
public:
    Accumlator(int n) : data(torch::zeros({n})), sum(torch::zeros({})), cnt(0) {}

    void add(torch::Tensor &x)
    {
        data += x;
        sum += x.sum();
        cnt += x.numel();
    }

    torch::Tensor mean()
    {
        return data / cnt;
    }

    float sum_item()
    {
        return sum.item<float>();
    }

private:
    torch::Tensor data;
    torch::Tensor sum;
    int cnt;
};

// float evaluate_accuracy(torch::nn::Module &net, torch::data::DataLoader<torch::data::datasets::MNIST> &data_loader);

void showBatchImage(torch::Tensor &image_batch);

torch::Tensor soft_max(torch::Tensor &x);

torch::Tensor cross_entropy(torch::Tensor &y_hat, torch::Tensor &y);

torch::Tensor net(torch::Tensor &x, torch::Tensor &W, torch::Tensor &b);

float accuracy(torch::Tensor &y_hat, torch::Tensor &y);

#endif // SOFT_MAX_H_