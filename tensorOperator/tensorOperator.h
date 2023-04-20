//
// Created by liumu on 23-4-3.
//

#ifndef DLRURTLE_CAPTURE2_TENSOROPERATOR_H_
#define DLRURTLE_CAPTURE2_TENSOROPERATOR_H_

#include <torch/torch.h>
#include <ATen/ATen.h>

class tensorOperator {
  public:
    tensorOperator() = default;
    ~tensorOperator() = default;
    void printTensor();
    void setTensor(torch::Tensor &t);
    void getTensor(torch::Tensor &t) const;
    void autoDiff();
    void prob();

private:
    torch::Tensor tensor;
};

#endif // DLRURTLE_CAPTURE2_TENSOROPERATOR_H_
