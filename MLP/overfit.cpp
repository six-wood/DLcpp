#include "overfit.h"

torch::Tensor train(torch::Tensor train_features, torch::Tensor train_labels,
                    torch::Tensor test_features, torch::Tensor test_labels,
                    float lr, int num_epochs, int batch_size,
                    std::vector<int> &x, std::vector<double> &y, std::vector<double> &y_hat)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

    AT_ASSERTM(train_features.size(0) == train_labels.size(0), "features and labels must have the same size");
    AT_ASSERTM(test_features.size(0) == test_labels.size(0), "features and labels must have the same size");

    torch::Tensor loss = torch::zeros({num_epochs, 1});
    auto train_data_set = MYDataset(train_features, train_labels).map(torch::data::transforms::Stack<>());
    auto test_data_set = MYDataset(test_features, test_labels).map(torch::data::transforms::Stack<>());
    auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_data_set), batch_size);
    auto test_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_data_set), batch_size);

    torch::nn::Sequential net({{"fc", torch::nn::Linear(torch::nn::LinearOptions(train_features.size(1), 1).bias(false))}});

    torch::optim::SGDOptions Soptions(lr);
    Soptions.momentum(0.0).weight_decay(1.0);

    auto l1 = [&](torch::Tensor &x) -> torch::Tensor
    {
        return x.abs().sum();
    };
    auto l2 = [&](torch::Tensor &x) -> torch::Tensor
    {
        return x.pow(2).sum();
    };

    auto optimizer = torch::optim::SGD(net->parameters(), Soptions);

    for (int i = 0; i < num_epochs; i++)
    {
        torch::Tensor loss_value = torch::zeros({1, 1}, options);
        if (i % 10 == 0)
            x.push_back(i);
        for (auto &batch : *train_iter)
        {
            optimizer.zero_grad();
            auto data = batch.data;
            auto target = batch.target;
            net->zero_grad();
            auto output = net->forward(data);
            loss_value = torch::mse_loss(output, target);
            loss_value = loss_value.sum() / output.size(0);
            loss_value.backward();
            optimizer.step();
        }

        auto train_pred = net->forward(train_features);
        auto train_loss = torch::mse_loss(train_pred, train_labels);

        if (i % 10 == 0)
            y.push_back(train_loss.sum().item<double>() / train_pred.size(0));

        auto test_pred = net->forward(test_features);
        auto test_loss = torch::mse_loss(test_pred, test_labels);

        if (i % 10 == 0)
            y_hat.push_back(test_loss.sum().item<double>() / test_pred.size(0));

        if (i % (num_epochs / 10) == 0)
            std::cout << "Epoch " << i << ". sum(loss_value) = " << loss_value.sum().item<double>() << std::endl;
    }
    std::cout << net->parameters() << std::endl;

    return net->parameters()[0];
}