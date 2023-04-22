import torch
import torchvision
from torchvision import transforms
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    net = []
    in_channels = 1
    for num_convs, out_channels in conv_arch:
        net.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *net,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10),
    )


net = vgg(conv_arch=conv_arch)

x = torch.randn(size=(1, 1, 244, 244))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, "output size:\t", x.shape)

trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
    ]
)

dataPath = "/home/liumu/code/DLcpp/linearNet/data/"

train_data_set = torchvision.datasets.FashionMNIST(
    root=dataPath, train=True, transform=trans, download=False
)

test_data_set = torchvision.datasets.FashionMNIST(
    root=dataPath, train=False, transform=trans, download=False
)

batch_size = 32

train_iter = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size, shuffle=True, num_workers=4
)
test_iter = torch.utils.data.DataLoader(
    test_data_set, batch_size=batch_size, shuffle=False, num_workers=4
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

loss = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device=device)

for epoch in range(10):
    batch_size = 0
    for x, y in train_iter:
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print(f"epoch {epoch}, loss {l}")
    with torch.no_grad():
        acc = 0
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc += (y_hat.argmax(axis=1) == y).float().sum().item()
        print(f"test acc {acc / len(test_data_set)}")
