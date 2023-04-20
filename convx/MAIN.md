# convolution
## convx
    torch::Tensor x = torch::ones({6, 8});
    cout << torch::arange(1, 8, 2) << endl;
    x.index_put_({"...", torch::tensor({2, 3, 4, 5})}, 0);
    torch::Tensor k = torch::tensor({-1, 1});
    cout << x << endl;
    cout << k << endl;
    torch::Tensor y = corr2d_1(x, k.reshape({1, 2}));
    cout << y << endl;

    torch::nn::Sequential net(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {1, 2}).stride(1).bias(false)));
    x = x.reshape({1, 1, 6, 8});
    k = k.reshape({1, 1, 1, 2});
    for (int i = 0; i < 10; ++i)
    {
        auto y_hat = net->forward(x);
        auto l = (y_hat - y).mul(y_hat - y).sum();
        net->zero_grad();
        l.backward();
        net[0]->named_parameters()["weight"].data() -= 0.03 * net[0]->named_parameters()["weight"].grad();
        if ((i + 1) % 2 == 0)
        {
            std::cout << "batch " << i + 1 << ", loss = " << l.data() << std::endl;
        }
    }
    cout << net->named_parameters()["0.weight"] << endl;

    auto convx2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {3, 3}).stride(1).padding(1).bias(false));
    convx2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {5, 3}).stride({2, 2}).padding({2, 1}).bias(false));
    y = convx2d->forward(x);
    cout << y << endl;

    auto pool2d = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1}));
    y = pool2d->forward(x);
    cout << y << endl;
## LeNet
    cout << "CUDnn: " << torch::cuda::cudnn_is_available() << endl;
    auto net = torch::nn::Sequential(nn::Conv2d(nn::Conv2dOptions(1, 6, {5, 5}).stride(1).padding(2).bias(false)),
                                     nn::Sigmoid(),
                                     nn::AvgPool2d(nn::AvgPool2dOptions({2, 2}).stride(2)),
                                     nn::Conv2d(nn::Conv2dOptions(6, 16, {5, 5}).stride(1).padding(0).bias(false)),
                                     nn::Sigmoid(),
                                     nn::AvgPool2d(nn::AvgPool2dOptions({2, 2}).stride(2)),
                                     nn::Flatten(),
                                     nn::Linear(nn::LinearOptions(16 * 5 * 5, 120).bias(false)),
                                     nn::Sigmoid(),
                                     nn::Linear(nn::LinearOptions(120, 84).bias(false)),
                                     nn::Sigmoid(),
                                     nn::Linear(nn::LinearOptions(84, 10).bias(false)),
                                     nn::LogSoftmax(1));
    auto X = torch::randn({1, 1, 28, 28});
    for (auto &layer : *net)
    {
        X = layer.forward(X);
        cout << layer.ptr()->name() << " output shape: " << X.sizes() << endl;
    }
    string dataPath = "/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw";
    auto train_dataset = make_shared<torch::data::datasets::MNIST>(dataPath, torch::data::datasets::MNIST::Mode::kTrain);
    auto test_dataset = make_shared<torch::data::datasets::MNIST>(dataPath, torch::data::datasets::MNIST::Mode::kTest);
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(train_dataset->map(torch::data::transforms::Stack<>()), 256);
    net->apply(init_weights);
    net->to(torch::kCUDA);
    torch::optim::SGD optimizer(net->parameters(), 0.1);
    clock_t start = clock(), end;
    for (int epoch = 0; epoch < 100; epoch++)
    {
        size_t batch_index = 0;
        for (auto &batch : *data_loader)
        {
            auto data = batch.data.to(torch::kCUDA);
            auto target = batch.target.to(torch::kCUDA);
            net->zero_grad();
            auto output = net->forward(data);
            auto loss = torch::nll_loss(output, target);
            loss.backward();
            optimizer.step();
            if (++batch_index % 64 == 0)
            {
                cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() << endl;
            }
        }
    }
    end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "Train Accuracy: " << evaluate_accuracy(net, train_dataset, torch::kCUDA) << endl;
    cout << "Test Accuracy: " << evaluate_accuracy(net, test_dataset, torch::kCUDA) << endl;    