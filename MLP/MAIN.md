# MLP
## show the picture
    auto options =
    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    torch::Tensor x = torch::arange(-8.0, 8.0, 0.1, options);
    torch::Tensor y = torch::relu(x);

    vector<float> a(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
    vector<float> b(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());

    y.backward(torch::ones_like(x), true);
    vector<float> c(x.grad().data_ptr<float>(),
    x.grad().data_ptr<float>() + x.grad().numel());

    torch::Tensor z = torch::sigmoid(x);
    vector<float> d(z.data_ptr<float>(), z.data_ptr<float>() + z.numel());

    x.grad().zero_();
    z.backward(torch::ones_like(x), true);
    vector<float> e(x.grad().data_ptr<float>(),
    x.grad().data_ptr<float>() + x.grad().numel());

    torch::Tensor o = torch::tanh(x);
    vector<float> f(o.data_ptr<float>(), o.data_ptr<float>() + o.numel());

    x.grad().zero_();
    o.backward(torch::ones_like(x), true);
    vector<float> g(x.grad().data_ptr<float>(),
    x.grad().data_ptr<float>() + x.grad().numel());

    plt::plot(a, b);
    plt::show();
    plt::plot(a, c);
    plt::show();
    plt::plot(a, d);
    plt::show();
    plt::plot(a, e);
    plt::show();
    plt::plot(a, f);
    plt::show();
    plt::plot(a, g);
    plt::show();
    return 0;
## mlp
    int n_in = 784, n_mid = 256, n_out = 10, batch_size = 64, epoch_size = 50;
    auto dataset = make_shared<torch::data::datasets::MNIST>("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw");
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset->map(torch::data::transforms::Stack<>()), batch_size);
    auto net = make_shared<MLP>(n_in, n_mid, n_out);
    torch::optim::SGD optimizer(net->parameters(), 0.01);

    long long int batch_index = 0;
    for (int epoch = 0; epoch < epoch_size; ++epoch)
    {
        for (auto &batch : *data_loader)
        {
            auto data = batch.data;
            auto target = batch.target;
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto loss = torch::nll_loss(output, target);
            loss.backward();
            optimizer.step();
            if (++batch_index % 500 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "/home/liumu/code/DLTurtle/model/mlp.pt");
            }
        }
    }
    cout << "over" << endl;
## overfit & model chose
    constexpr int max_degree = 20;
    constexpr int n_train = 800;
    constexpr int n_test = 200;

    torch::Tensor true_w = torch::zeros(max_degree);
    float temp[] = {5.0, 1.2, -3.4, 5.6};
    memcpy(true_w.data_ptr(), temp, sizeof(temp));

    // Generate data
    true_w = true_w.reshape({max_degree, 1});
    torch::Tensor x_train = torch::randint(-99999, 99999, {n_train + n_test, 1}) / 100000.0;
    torch::Tensor features = x_train.pow(torch::arange(max_degree));

    for (int i = 0; i < max_degree; i++)
    {
        int factorial = 1;
        if (i != 0)
            factorial = tgamma(i);
        for (int row = 0; row < (n_train + n_test); row++)
        {
            features[row][i] /= factorial;
        }
    }

    torch::Tensor labels = torch::mm(features, true_w) + torch::randn({n_train + n_test, 1}, torch::kFloat32) * 0.1;

    cout << "features size: " << features.size(0) << " " << features.size(1) << endl;
    cout << "true_w size: " << true_w.size(0) << " " << true_w.size(1) << endl;
    // Train and test

    auto train_data = features.index({torch::indexing::Slice(0, n_train, torch::indexing::None), torch::indexing::Slice(0, 4, torch::indexing::None)}).clone();
    auto train_label = labels.index({torch::indexing::Slice(0, n_train, torch::indexing::None), torch::indexing::Slice(0, 4, torch::indexing::None)}).clone();
    auto test_data = features.index({torch::indexing::Slice(n_train, n_train + n_test, torch::indexing::None), torch::indexing::Slice(0, 4, torch::indexing::None)}).clone();
    auto test_label = labels.index({torch::indexing::Slice(n_train, n_train + n_test, torch::indexing::None), torch::indexing::Slice(0, 4, torch::indexing::None)}).clone();

    cout << "train_data size: " << train_data.size(0) << " " << train_data.size(1) << endl;
    cout << "train_label size: " << train_label.size(0) << " " << train_label.size(1) << endl;
    cout << "test_data size: " << test_data.size(0) << " " << test_data.size(1) << endl;
    cout << "test_label size: " << test_label.size(0) << " " << test_label.size(1) << endl;

    std::vector<int> x;
    std::vector<double> y, y_hat;

    auto w = train(train_data, train_label, test_data, test_label, true_w, 0.01, 4000, 10, x, y, y_hat);

    plt::semilogy(x, y, "b");
    plt::semilogy(x, y_hat, "r");
    plt::title("loss(r:test b:train)");
    plt::legend();
    plt::show();
## weight decay
    constexpr int n_w = 200;
    constexpr int n_train = 20;
    constexpr int n_test = 200;
    constexpr int n_choose = 200;

    torch::Tensor true_w = 0.01 * torch::ones(n_w);
    torch::Tensor true_b = torch::tensor({0.05});

    // Generate data
    torch::Tensor x_train = torch::randint(-99999, 99999, {n_train + n_test, n_w}) / 100000.0;
    torch::Tensor features = x_train.clone();
    features = features.reshape({n_train + n_test, n_w});
    true_w = true_w.reshape({n_w, 1});
    true_b = true_b.reshape({1, 1});
    torch::Tensor labels = torch::mm(features, true_w) + true_b + torch::randn({n_train + n_test, 1}, torch::kFloat32) * 0.1;

    cout << "features size: " << features.size(0) << " " << features.size(1) << endl;
    cout << "true_w size: " << true_w.size(0) << " " << true_w.size(1) << endl;
    // Train and test

    auto train_data = features.index({torch::indexing::Slice(0, n_train, torch::indexing::None), torch::indexing::Slice(0, n_choose, torch::indexing::None)}).clone();
    auto train_label = labels.index({torch::indexing::Slice(0, n_train, torch::indexing::None), torch::indexing::Slice(0, n_choose, torch::indexing::None)}).clone();
    auto test_data = features.index({torch::indexing::Slice(n_train, n_train + n_test, torch::indexing::None), torch::indexing::Slice(0, n_choose, torch::indexing::None)}).clone();
    auto test_label = labels.index({torch::indexing::Slice(n_train, n_train + n_test, torch::indexing::None), torch::indexing::Slice(0, n_choose, torch::indexing::None)}).clone();

    cout << "train_data size: " << train_data.size(0) << " " << train_data.size(1) << endl;
    cout << "train_label size: " << train_label.size(0) << " " << train_label.size(1) << endl;
    cout << "test_data size: " << test_data.size(0) << " " << test_data.size(1) << endl;
    cout << "test_label size: " << test_label.size(0) << " " << test_label.size(1) << endl;

    std::vector<int> x;
    std::vector<double> y, y_hat;
    if (!x.empty())
        std::vector<int>().swap(x);
    if (!y.empty())
        std::vector<double>().swap(y);
    if (!y_hat.empty())
        std::vector<double>().swap(y_hat);

    auto w = train(train_data, train_label, test_data, test_label, 0.01, 4000, 5, x, y, y_hat);

    plt::semilogy(x, y, "b");
    plt::semilogy(x, y_hat, "r");
    plt::title("loss(r:test b:train)");
    plt::legend();
    plt::show();
## dropout 
    torch::Tensor x = torch::arange(0, 10, 1, torch::kFloat32);
    std::cout << x << std::endl;
    std::cout << dropout_layer(x, 0.5) << std::endl;
    std::cout << dropout_layer(x, 0.0) << std::endl;
    std::cout << dropout_layer(x, 1.0) << std::endl;

    constexpr int input_size = 784;
    constexpr int hidden_size1 = 64;
    constexpr int hidden_size2 = 32;
    constexpr int output_size = 10;
    constexpr int batch_size = 256;

    auto net = make_shared<mlp_dropout>(input_size, hidden_size1, hidden_size2, output_size); // 0.2, 0.5
    auto train_dataset = torch::data::datasets::MNIST("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw")
                             .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw", torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), 256);
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), 256);

    vector<int> x_axis;
    vector<double> y, y_hat;
    if (!x_axis.empty())
        std::vector<int>().swap(x_axis);
    if (!y.empty())
        std::vector<double>().swap(y);
    if (!y_hat.empty())
        std::vector<double>().swap(y_hat);

    auto optimizer = torch::optim::SGD(net->parameters(), 0.1);
    for (int epoch = 0; epoch < 100; epoch++)
    {
        size_t batch_index = 0;
        torch::Tensor train_loss = torch::zeros({1});
        torch::Tensor test_loss = torch::zeros({1});
        for (auto train_data : *train_data_loader)
        {
            auto x = train_data.data;
            auto y = train_data.target;
            auto y_hat = net->forward(x);
            auto loss = torch::nll_loss(y_hat, y);
            train_loss += loss;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            if (++batch_index % 64 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
        y.push_back(train_loss.item<float>());
        x_axis.push_back(epoch);
        for (auto test_data : *test_data_loader)
        {
            auto x = test_data.data;
            auto y = test_data.target;
            auto y_hat = net->forward(x);
            auto loss = torch::nll_loss(y_hat, y);
            test_loss += loss;
        }
        y_hat.push_back(test_loss.item<float>());
    }

    plt::plot(x_axis, y_hat, "r");
    plt::plot(x_axis, y, "b");
    plt::title("loss: test_loss:red, train_loss:blue");
    plt::legend();
    plt::show();

## sqeuential
    torch::Tensor x = torch::arange(0, 10, 1, torch::kFloat32);
    std::cout << x << std::endl;
    std::cout << dropout_layer(x, 0.5) << std::endl;
    std::cout << dropout_layer(x, 0.0) << std::endl;
    std::cout << dropout_layer(x, 1.0) << std::endl;

    constexpr int input_size = 784;
    constexpr int hidden_size1 = 64;
    constexpr int hidden_size2 = 32;
    constexpr int output_size = 10;
    constexpr int batch_size = 256;

    auto net = make_shared<mlp_dropout>(input_size, hidden_size1, hidden_size2, output_size); // 0.2, 0.5
    torch::nn::Sequential net2(
        torch::nn::Flatten(),
        torch::nn::Linear(input_size, hidden_size1),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.2),
        torch::nn::Linear(hidden_size1, hidden_size2),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(hidden_size2, output_size),
        torch::nn::LogSoftmax(1));
    auto train_dataset = torch::data::datasets::MNIST("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw")
                             .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw", torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), 256);
    auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), 256);

    vector<int> x_axis;
    vector<double> y, y_hat;
    if (!x_axis.empty())
        std::vector<int>().swap(x_axis);
    if (!y.empty())
        std::vector<double>().swap(y);
    if (!y_hat.empty())
        std::vector<double>().swap(y_hat);

    auto optimizer = torch::optim::SGD(net2->parameters(), 0.1);
    for (int epoch = 0; epoch < 100; epoch++)
    {
        size_t batch_index = 0;
        torch::Tensor train_loss = torch::zeros({1});
        torch::Tensor test_loss = torch::zeros({1});
        for (auto train_data : *train_data_loader)
        {
            auto x = train_data.data;
            auto y = train_data.target;
            auto y_hat = net2->forward(x);
            auto loss = torch::nll_loss(y_hat, y);
            train_loss += loss;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            if (++batch_index % 64 == 0)
            {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
        y.push_back(train_loss.item<float>());
        x_axis.push_back(epoch);
        for (auto test_data : *test_data_loader)
        {
            auto x = test_data.data;
            auto y = test_data.target;
            auto y_hat = net2->forward(x);
            auto loss = torch::nll_loss(y_hat, y);
            test_loss += loss;
        }
        y_hat.push_back(test_loss.item<float>());
    }

    plt::plot(x_axis, y_hat, "r");
    plt::plot(x_axis, y, "b");
    plt::title("loss: test_loss:red, train_loss:blue");
    plt::legend();
    plt::show();