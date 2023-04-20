#ifndef USEGPU_H_
#define USEGPU_H_

#include <iostream>
#include <torch/torch.h>

torch::Device try_gpu(int i = 0)
{
    std::cout << "calling try_gpu() ..." << std::endl;
    if (torch::cuda::device_count() >= (i + 1))
    {
        return torch::Device(torch::kCUDA, i);
    }

    std::cout << "no gpu found!" << std::endl;
    return torch::Device("cpu");
}

std::vector<torch::Device> try_all_gpus()
{
    std::cout << "calling try_all_gpus() ..." << std::endl;
    std::vector<torch::Device> devices;

    if (torch::cuda::device_count())
    {
        int c = torch::cuda::device_count();
        std::cout << "total gpu count is : " << torch::cuda::device_count() << std::endl;
        for (int i = 0; i < c; i++)
        {
            devices.push_back(torch::Device(torch::kCUDA, i));
        }
    }
    else
    {
        devices.push_back(torch::Device("cpu"));
    }
    return devices;
}
#endif /*USEGPU_H_*/