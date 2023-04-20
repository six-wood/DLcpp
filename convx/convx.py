import torch
import torch.nn as nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
k = torch.tensor([[0, 1], [2, 3]])

print(corr2d(x, k))


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


x = torch.ones((6, 8))
x[:, 2:6] = 0

k = torch.tensor([[1, -1]])
y = corr2d(x, k)
print(y)
z = corr2d(x, k.T)
print(z)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
x = x.reshape((1, 1, 6, 8))
y = y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    y_hat = conv2d(x)
    l = ((y_hat - y) ** 2).sum()
    conv2d.zero_grad()
    l.backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f"batch {i+1}, loss {l.item():.3f}")
print(conv2d.weight.data.reshape((1, 2)))


def comp_conv2d(conv2d, x):
    x = x.reshape((1, 1) + x.shape)
    y = conv2d(x)
    return y.reshape(y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
x = torch.rand(8, 8)
print(comp_conv2d(conv2d, x).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
conv2d = nn.Conv2d(1, 2, kernel_size=(3, 5), padding=(1, 2), stride=(2, 2))
x = x.reshape((1, 1) + x.shape)
y = conv2d(x)
print(y)


def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i : i + p_h, j : j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i : i + p_h, j : j + p_w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))

pool2d_torch = nn.MaxPool2d([2, 2], stride=1)
print(pool2d_torch(X.reshape((1, 1) + X.shape)))
