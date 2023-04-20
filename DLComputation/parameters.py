import torch
import torch.nn as nn
import numpy as np
import pandas as pd

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net[2].state_dict())
print(net[2].bias)
print(net[2].bias.data)

print(net[2].bias.grad == None)

print(*[(name, param.shape) for name, param in net[2].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()["2.bias"].data)


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block {i}", block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)

print(rgnet[0][1][0].bias.data)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, val=0)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, val=1)
        nn.init.constant_(m.bias, val=0)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, val=42)


net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0], net[2].weight.data[0])
print(net[2].weight.data[0], net[2].bias.data[0])


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()])
        nn.init.uniform_(m.weight, -10, 10)
        nn.init.zeros_(m.bias)


net.apply(my_init)
print(net[0].weight.data[0], net[0].bias.data[0])

net[0].weight.data[:] += 1
print(net[0].weight.data[0])

x = np.array([[1, 2, 3], [4, 5, 6]])
x_t = torch.tensor(x)
print(x_t.matmul(x_t.T))
print(x_t.T.matmul(x_t))

shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1)
)
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data)
print(net[2].weight.data[0] == net[4].weight.data[0])
