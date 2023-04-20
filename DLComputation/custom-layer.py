import torch
import torch.nn as nn
import torch.nn.functional as F


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
y = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
print(y)
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(y.mean().item())


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(
            torch.randn(
                units,
            )
        )

    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
x = torch.randn(1, 64)
y = net(x)
print(y)
print(F.relu(torch.randn((2, 3))))

torch.save(y, "x-file")
y2 = torch.load("x-file")
print(y2 - y)
torch.save([y, y], "x-files")
y3, y4 = torch.load("x-files")
mdict = {"y3": y3, "y4": y4}
torch.save(mdict, "x-dict")
mdict2 = torch.load("x-dict")
print(mdict2)

torch.save(net.state_dict(), "MyLinear.params")
clone = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
clone.load_state_dict(torch.load("MyLinear.params"))
print(clone.eval())

y_clone = clone(x)
print(y == y_clone)

torch.save(net, "net-all")
clone_all = torch.load("net-all")
y_clone_all = clone_all(x)
print(y_clone == y_clone_all)
