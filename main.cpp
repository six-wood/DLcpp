#include <ctime>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <matplot/matplotlibcpp.h>

namespace plt = matplotlibcpp;
using namespace std;
using namespace torch;

int main()
{
    string dataPath = "/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw";
    auto train_data_set = torch::data::datasets::MNIST(dataPath)
                              .map(torch::data::transforms::Stack<>());
    nn::Sequential net = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(1, 96, 11).stride(4).padding(2)), nn::ReLU(),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)),
        nn::Conv2d(nn::Conv2dOptions(96, 256, 5).stride(1).padding(2)), nn::ReLU(),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)),
        nn::Conv2d(nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
        nn::Conv2d(nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
        nn::Conv2d(nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)), nn::Flatten(),
        nn::Linear(9216, 4096), nn::ReLU(), nn::Dropout(0.5),
        nn::Linear(4096, 4096), nn::ReLU(), nn::Dropout(0.5),
        nn::Linear(4096, 1000));
}