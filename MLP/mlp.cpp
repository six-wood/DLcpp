#include "mlp.h"

MLP::MLP(int n_in, int n_mid, int n_out)
{
  fc1 = register_module("fc1", torch::nn::Linear(n_in, n_mid));
  fc2 = register_module("fc2", torch::nn::Linear(n_mid, n_out));
}

torch::Tensor MLP::forward(torch::Tensor x)
{
  x = fc1->forward(x.reshape({x.size(0), 784}));
  x = torch::relu(x);
  x = fc2->forward(x);
  x = torch::log_softmax(x, /*dim=*/1);
  return x;
}