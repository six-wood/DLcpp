#ifndef MLP_H_
#define MLP_H_

#include <torch/torch.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>

class MLP : public torch::nn::Module {
public:
  MLP(int n_in, int n_mid, int n_out);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

#endif // MLP_H_
