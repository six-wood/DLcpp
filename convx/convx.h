#ifndef CONVX_H_
#define CONVX_H_

#include <torch/torch.h>

torch::Tensor convx(torch::Tensor x, torch::Tensor w, torch::Tensor b, int stride, int padding);

class Convx2D : public torch::nn::Module
{
public:
    Convx2D(int in_channels, int out_channels, int kernel_size, int stride, int padding);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
};

torch::Tensor corr2d(torch::Tensor x, torch::Tensor k);

torch::Tensor corr2d_1(torch::Tensor X, torch::Tensor K)
{
    int h = K.size(0);
    int w = K.size(1);

    torch::Tensor Y = torch::zeros({X.size(0) - h + 1, X.size(1) - w + 1});
    for (int i = 0; i < Y.size(0); i++)
    {
        for (int j = 0; j < Y.size(1); j++)
        {
            Y[i][j] = (X.index({torch::indexing::Slice(i, i + h), torch::indexing::Slice(j, j + w)}) * K).sum();
        }
    }

    return Y;
}

void init_weights(torch::nn::Module &m)
{
    if (typeid(m) == typeid(torch::nn::Conv2dImpl))
    {
        torch::nn::init::xavier_uniform_(m.as<torch::nn::Conv2dImpl>()->weight);
    }
}

double evaluate_accuracy(torch::nn::Sequential &net, std::shared_ptr<torch::data::datasets::MNIST> &dataset, torch::Device device)
{
    int64_t n = 0, correct = 0;
    net->eval();
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset->map(torch::data::transforms::Stack<>()), 256);
    for (auto &batch : *data_loader)
    {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        auto output = net->forward(data);
        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().item<int64_t>();
        n += data.size(0);
    }
    return (double)correct / (double)n;
}

#endif /*CONVX_H_*/