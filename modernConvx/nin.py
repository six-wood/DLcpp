import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
    return blk

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    # The global average pooling layer automatically sets the kernel size
    # to the height and width of the input
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output
    # with a shape of (batch size, 10)
    nn.Flatten())

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

dataPath = "/home/liumu/code/DLcpp/linearNet/data/"

trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
    ]
)

train_data_set = torchvision.datasets.FashionMNIST(
    root=dataPath, train=True, transform=trans, download=False
)

test_data_set = torchvision.datasets.FashionMNIST(
    root=dataPath, train=False, transform=trans, download=False
)

batch_size = 64
train_iter = torch.utils.data.DataLoader(
    train_data_set, batch_size=batch_size, shuffle=True, num_workers=4
)
test_iter = torch.utils.data.DataLoader(
    test_data_set, batch_size=batch_size, shuffle=False, num_workers=4
)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

loss = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device)

for epoch in range(5):
    acc = 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
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
