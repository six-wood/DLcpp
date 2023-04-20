#include "convx.h"

torch::Tensor convx(torch::Tensor x, torch::Tensor w, torch::Tensor b, int stride, int padding)
{
    torch::Tensor out = torch::conv2d(x, w, b, {stride, stride}, {padding, padding});
    return out;
}

torch::Tensor corr2d(torch::Tensor x, torch::Tensor k)
{
    torch::Tensor w = k.flip({0, 1});
    torch::Tensor b = torch::zeros({1});
    return convx(x, w, b, 1, 0);
}