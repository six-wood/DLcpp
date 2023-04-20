# deep learning computation
## model construction
    constexpr int input_size = 784;
    constexpr int hidden_size1 = 256;
    constexpr int hidden_size2 = 256;
    constexpr int num_classes = 10;
    constexpr int64_t kNumEpochs = 100;
    constexpr int64_t kBatchSize = 64;

    auto net = std::make_shared<mlp_computation>(input_size, hidden_size1, hidden_size2, num_classes);
    torch::Tensor x = torch::randn({1, input_size});
    auto y = net->forward(x);
    cout << y << endl;

    torch::nn::Sequential net2(
        torch::nn::Linear(input_size, hidden_size1),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(hidden_size1, hidden_size2),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.2),
        torch::nn::Linear(hidden_size2, num_classes),
        torch::nn::LogSoftmax(1));

    auto y2 = net2->forward(x);
    cout << y2 << endl;

    auto net3 = std::make_shared<FixedMLP>(input_size, hidden_size1, hidden_size2, num_classes);
    auto y3 = net3->forward(x);
    cout << y3 << endl;

    auto net4 = std::make_shared<NestMLP>();
    auto y4 = net4->forward(x);
    cout << y4 << endl;
## parameters
    constexpr int input_size = 784;
    constexpr int hiden_size1 = 256;
    constexpr int hiden_size2 = 10;
    constexpr int output_size = 10;
    constexpr int batch_size = 256;

    torch::nn::Sequential net(
        torch::nn::Linear(input_size, hiden_size1),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(hiden_size1, hiden_size2),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.2),
        torch::nn::Linear(hiden_size2, output_size),
        torch::nn::LogSoftmax(1));

    cout << net[6]->name() << endl;
    auto od = net[6]->named_parameters();
    for (auto &p : od)
    {
        cout << p.key() << endl;
        cout << p.value() << endl;
        cout << "<<<<....>>>>" << endl;
    }
    auto b = od.find("bias");
    cout << b->data() << endl;

    auto w = od.find("weight");
    cout << w->data() << endl;
    cout << "======================" << endl;
    cout << w->grad() << endl;
    cout << "======================" << endl;

    auto od1 = net->named_parameters();
    for (const auto &p : od1)
    {
        cout << p.key() << endl;
        cout << p.value() << endl;
        cout << "<<<...>>>" << endl;
    }

    torch::nn::Sequential rgnet(block2(), torch::nn::Linear(4, 1));
    rgnet->push_back(torch::nn::ReLU());
    auto y = rgnet->forward(torch::randn({1, 4}));
    cout << y << endl;
    cout << rgnet << endl;
    for (int i = 0; i < rgnet->size(); i++)
    {
        cout << rgnet[i]->name() << endl;
    }

    auto od2 = rgnet[0]->named_parameters();
    for (const auto &p : od2)
    {
        cout << p.key() << endl;
        cout << p.value() << endl;
        cout << "<<<...>>>" << endl;
    }
    auto linear = make_shared<MyLinear>(4, 1);
    auto y1 = linear->forward(torch::randn({1, 4}));
    auto od3 = linear->named_parameters();
    for (const auto &p : od3)
    {
        cout << p.key() << endl;
        cout << p.value() << endl;
        cout << "<<<...>>>" << endl;
    }
    cout << y1 << endl;

    net = torch::nn::Sequential(MyLinear(64, 8), MyLinear(8, 1));
    auto y2 = net->forward(torch::randn({1, 64}));
    cout << y2 << endl;

    torch::save(y2, "y-file");
    torch::Tensor y3;
    torch::load(y3, "y-file");
    cout << (y2 == y3) << endl;
    cout << y3.device() << endl;
## use gpu
    torch::Tensor x = torch::randn({1, 2}, torch::kCUDA);
    cout << x.device() << endl;
    torch::Tensor y = torch::randn({1, 2});
    y.to(torch::kCUDA);
    cout << y.device() << endl;

    torch::Device device_gpu = torch::kCUDA;
    torch::Device device_cpu = torch::kCPU;
    device_gpu.set_index(0);

    x = torch::randn({1, 2}, try_gpu());
    cout << x.device() << endl;
    auto z = x.to(torch::Device(torch::kCUDA, 0));
    cout << x + z << endl;

    auto net = torch::nn::Sequential(
        torch::nn::Linear(2, 2),
        torch::nn::Linear(2, 2));
    net->to(torch::kCUDA);
    z = net->forward(x);
    cout << z << endl;

    clock_t start, end;
    start = clock();
    torch::Tensor a = torch::randn({10000, 10000});
    torch::Tensor b = torch::randn({10000, 10000});
    torch::Tensor c = a.matmul(b);
    end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "duration: " << duration << endl;

    start = clock();
    a = torch::randn({10000, 10000}, torch::kCUDA);
    b = torch::randn({10000, 10000}, torch::kCUDA);
    c = a.matmul(b);
    end = clock();
    duration = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "duration: " << duration << endl;