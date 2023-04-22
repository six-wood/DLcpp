# modernCOnvx
## AlexNet
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
    auto resize = torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({244, 244})).mode(torch::kNearest);

    nn::Sequential net = nn::Sequential(
        nn::Conv2d(nn::Conv2dOptions(1, 96, 11).stride(4).padding(1)), nn::ReLU(),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)),
        nn::Conv2d(nn::Conv2dOptions(96, 256, 5).stride(1).padding(2)), nn::ReLU(),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)),
        nn::Conv2d(nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
        nn::Conv2d(nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
        nn::Conv2d(nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
        nn::MaxPool2d(nn::MaxPool2dOptions(3).stride(2)), nn::Flatten(),
        nn::Linear(9216, 4096), nn::ReLU(), nn::Dropout(0.5),
        nn::Linear(4096, 4096), nn::ReLU(), nn::Dropout(0.5),
        nn::Linear(4096, 10));

    auto x = torch::randn({1, 1, 28, 28});
    x = nn::functional::interpolate(x, resize);
    for (auto &layer : *net)
    {
        x = layer.forward(x);
        cout << layer.ptr()->name() << " " << x.sizes() << endl;
    }

    auto optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(0.001));

    auto loss = nn::CrossEntropyLoss();

    auto device_type = torch::kCUDA;

    net->to(device_type);

    for (int epoch = 0; epoch < 10; epoch++)
    {
        int batch_idx = 0;
        for (auto &batch : *train_data_loader)
        {
            auto data = batch.data.to(device_type);
            auto target = batch.target.to(device_type);
            data = nn::functional::interpolate(data, resize);
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto l = loss(output, target);
            l.backward();
            optimizer.step();
            if (++batch_idx % 100 == 0)
            {
                cout << "Train Epoch: " << epoch << " [" << batch_idx * batch_size << "/" << train_data_set.size().value() << " (" << 100. * batch_idx * batch_size / train_data_set.size().value() << "%)]\tLoss: " << l.item<float>() << endl;
            }
        }
        double acc = 0;
        for (auto &batch : *test_data_loader)
        {
            auto data = batch.data.to(device_type);
            auto target = batch.target.to(device_type);
            data = nn::functional::interpolate(data, resize);
            auto output = net->forward(data);
            auto pred = output.argmax(1);
            acc += pred.eq(target).sum().item<double>();
        }
        acc /= test_data_set.size().value();
        cout << "Test set: Accuracy: " << acc << endl;
    }
##VGG
    vector<vector<int>> conv_arch = {{1, 16}, {1, 32}, {2, 32}, {2, 128}, {2, 128}};
    auto net = myVGG(conv_arch);

    torch::Tensor x = torch::randn({1, 1, 224, 224});

    auto resize = torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({244, 244})).mode(torch::kNearest);

    auto optimizer = torch::optim::Adam(net.parameters(), torch::optim::AdamOptions(0.001));

    auto loss = nn::CrossEntropyLoss();

    auto device_type = torch::kCUDA;

    net.to(device_type);

    for (int epoch = 0; epoch < 10; epoch++)
    {
        int batch_idx = 0;
        for (auto &batch : *train_data_loader)
        {
            auto data = batch.data.to(device_type);
            auto target = batch.target.to(device_type);
            data = nn::functional::interpolate(data, resize);
            optimizer.zero_grad();
            auto output = net.forward(data);
            auto l = loss(output, target);
            l.backward();
            optimizer.step();
            if (++batch_idx % 100 == 0)
            {
                cout << "Train Epoch: " << epoch << " [" << batch_idx * batch_size << "/" << train_data_set.size().value() << " (" << 100. * batch_idx * batch_size / train_data_set.size().value() << "%)]\tLoss: " << l.item<float>() << endl;
            }
        }
        double acc = 0;
        for (auto &batch : *test_data_loader)
        {
            auto data = batch.data.to(device_type);
            auto target = batch.target.to(device_type);
            data = nn::functional::interpolate(data, resize);
            auto output = net.forward(data);
            auto pred = output.argmax(1);
            acc += pred.eq(target).sum().item<double>();
        }
        acc /= test_data_set.size().value();
        cout << "Test set: Accuracy: " << acc << endl;
    }