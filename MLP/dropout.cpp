#include "dropout.h"

using namespace std;
using namespace torch;

Tensor dropout_layer(Tensor x, float dropout)
{
    if (dropout == 1.0)
    {
        return torch::zeros_like(x);
    }
    if (dropout == 0.0)
    {
        return x.clone();
    }
    Tensor mask = torch::bernoulli(torch::ones_like(x) * (1 - dropout));
    return mask * x / (1.0 - dropout);
}

Tensor mlp_dropout::forward(Tensor x)
{
    x = fc1->forward(x.reshape({x.size(0), -1}));
    x = torch::relu(x);
    x = dropout_layer(x, 0.2);
    x = fc2->forward(x);
    x = torch::relu(x);
    x = dropout_layer(x, 0.5);
    x = fc3->forward(x);
    x = torch::log_softmax(x, 1);
    return x;
}