#include "softMax.h"

void showBatchImage(torch::Tensor &image_batch)
{
    cv::Mat img = cv::Mat(28 * image_batch.size(0), 28, CV_32FC1, (void *)(image_batch.data_ptr())).clone();
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("image", img);
    cv::waitKey(0);
}

inline torch::Tensor soft_max(torch::Tensor &x)
{
    torch::Tensor x_exp = torch::exp(x);
    return x_exp / x_exp.sum(1, true);
}

inline torch::Tensor cross_entropy(torch::Tensor &y_hat, torch::Tensor &y)
{
    return -torch::log(y_hat.index({torch::indexing::Slice(), y}));
}

inline torch::Tensor net(torch::Tensor &x, torch::Tensor &W, torch::Tensor &b)
{
    torch::Tensor y_hat = torch::matmul(x.reshape({-1, W.size(0)}), W) + b;
    return soft_max(y_hat);
}

float accuracy(torch::Tensor &y_hat, torch::Tensor &y)
{
    return (y_hat.argmax(1) == y).sum().item<float>();
}

// float evaluate_accuracy(torch::nn::Module &net, torch::data::DataLoader<torch::data::datasets::MNIST> &data_loader)
// {
    
//     float sum = 0;
//     int cnt = 0;
//     for (auto &batch : data_loader)
//     {
//         torch::Tensor X = batch.data;
//         torch::Tensor y = batch.target;
//         sum += (net(X).argmax(1) == y).sum().item<float>();
//         cnt += y.size(0);
//     }
//     return sum / cnt;
// }