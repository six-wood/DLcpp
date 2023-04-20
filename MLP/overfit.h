#ifndef OVERFIT_H_
#define OVERFIT_H_

#include <torch/torch.h>

class MYDataset : public torch::data::Dataset<MYDataset>
{
public:
    explicit MYDataset(torch::Tensor data, torch::Tensor target)
        : data_(std::move(data)), target_(std::move(target)) {}
    torch::data::Example<> get(size_t index) override
    {
        return {data_[index], target_[index]};
    }
    torch::optional<size_t> size() const override
    {
        return data_.size(0);
    }

private:
    torch::Tensor data_, target_;
};

torch::Tensor train(torch::Tensor train_features, torch::Tensor train_labels,
                    torch::Tensor test_features, torch::Tensor test_labels,
                    float lr, int num_epochs, int batch_size,
                    std::vector<int> &x, std::vector<double> &y, std::vector<double> &y_hat);

#endif /* OVERFIT_H_ */