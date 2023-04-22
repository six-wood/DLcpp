#include <ctime>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <matplot/matplotlibcpp.h>
#include <modernConvx/modernConvx.h>

namespace plt = matplotlibcpp;
using namespace std;
using namespace torch;

string dataPath = "/home/liumu/code/DLcpp/linearNet/data/FashionMNIST/raw";
int batch_size = 64;

auto train_data_set = torch::data::datasets::MNIST(dataPath, torch::data::datasets::MNIST::Mode::kTrain)
                          .map(torch::data::transforms::Stack<>());

auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    train_data_set, batch_size);

auto test_data_set = torch::data::datasets::MNIST(dataPath, torch::data::datasets::MNIST::Mode::kTest)
                         .map(torch::data::transforms::Stack<>());

auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    test_data_set, batch_size);

int main()
{

}