import torch
import torch.nn as nn

torch.device("cpu"), torch.device("cuda:0")
print(torch.cuda.is_available())
print(torch.cuda.device_count())


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():  # @save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1, 2, 3])
print(x.device)

x = x.to(device=try_gpu(0))
print(x.device)
z = x.to(device=try_gpu(0))

print(x + z)
