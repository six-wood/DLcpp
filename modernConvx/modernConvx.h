#ifndef MODERHCONVX_H_
#define MODERNCONVX_H_

#include <torch/torch.h>
#include <vector>

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                             int64_t stride = 1, int64_t padding = 0, bool with_bias = false)
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;
}

inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride)
{
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    return maxpool_options;
}

torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm);

class VGGImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential features_{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier;

public:
    VGGImpl(std::vector<int> &cfg, int num_classes = 1000, bool batch_norm = false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(VGG);

class vggBlock : public torch::nn::Module
{
private:
    torch::nn::Sequential seq{nullptr};

public:
    vggBlock(int num_convx, int in_channels, int out_channels);
    torch::Tensor forward(torch::Tensor x);
};

class myVGG : public torch::nn::Module
{
private:
    torch::nn::Sequential seq{nullptr};

public:
    myVGG(std::vector<std::vector<int>> &conv_arch);
    torch::Tensor forward(torch::Tensor x);
};


#endif /*MODERNCONVX_H_*/