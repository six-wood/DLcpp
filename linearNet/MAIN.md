# using guide
## linear regression

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // Print the tensor.
    torch::Tensor w = torch::tensor({2.0f, -3.4f}, options);

    torch::Tensor b = torch::tensor({4.2f}, options);

    lineaRegression linea_regression(w, b, 1000);
    linea_regression.generateData();

    torch::Tensor features, labels;
    coro_t::pull_type data_iter([&](coro_t::push_type &yield)
                                { linea_regression.dataIter(10, features, labels, yield); });

    torch::Tensor w_t = torch::normal(0, 0.01, {2, 1});
    w_t.set_requires_grad(true);
    torch::Tensor b_t = torch::zeros({1, 1});
    b_t.set_requires_grad(true);

    torch::Tensor y_hat, l, features_true, labels_true;
    std::vector<torch::Tensor> parmas;
    linea_regression.getFeature(features_true);
    linea_regression.getLabels(labels_true);

    for (int i = 0; i < 3; i++)
    {
        for (auto &data : data_iter)
        {
        y_hat = linea_regression.linearRegression(features, w_t, b_t);
        l = linea_regression.squared_loss(y_hat, labels).sum();
        l.backward();
        parmas = {w_t, b_t};
        linea_regression.sgd(parmas, 0.03, 10);
        }

        torch::Tensor lables_train = linea_regression.linearRegression(features_true, w_t, b_t);
        torch::Tensor train_loos = linea_regression.squared_loss(lables_train, labels_true).mean();
        std::cout << "epoch " << i << ", loss " << train_loos.item<float>() << std::endl;
    }

    auto dataset = LineaRegressionDataset(features_true, labels_true).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), 10);
    auto net = std::make_shared<LineaRegressionModule>(2, 1);
    torch::optim::SGD optimizer(net->parameters(), 0.03);

    std::size_t count = 0;
    std::vector<float> loss_v;
    loss_v.reserve(50);

    for (auto &batch : *data_loader)
    {
        auto data = batch.data;
        auto target = batch.target;
        data = data.to(torch::kF32);
        target = target.to(torch::kF32);
        optimizer.zero_grad();
        auto output = net->forward(data);
        auto loss = torch::mse_loss(output, target);

        loss.backward();
        optimizer.step();
        ++count;
        auto loss_value = loss.mean().item<float>();
        loss_v.emplace_back(loss_value);
        std::cout << "epoch " << count << ", loss " << loss_value << std::endl;
    }
    plt::plot(loss_v);
    plt::title("Loss");
    plt::xlim(0, 50);
    plt::xlabel("epoch");
    plt::ylabel("loss");
    plt::show();
## soft max
    int num_inputs = 28 * 28;
    int num_outputs = 10;

    torch::Tensor W = torch::randn({num_inputs, num_outputs}, torch::requires_grad());
    torch::Tensor b = torch::randn({num_outputs}, torch::requires_grad());

    torch::Tensor y = torch::tensor(at::ArrayRef<int>({0, 2}));
    torch::Tensor y_hat = torch::tensor(at::ArrayRef<float>({0.1, 0.3, 0.6, 0.3, 0.2, 0.5}));
    y_hat = y_hat.reshape({2, 3});

    auto sm = make_shared<softMax>(num_inputs, num_outputs);

    auto train_dataset = std::make_shared<torch::data::datasets::MNIST>("/home/liumu/code/DLTurtle/linearNet/data/FashionMNIST/raw");
    torch::Tensor image = train_dataset->images();
    torch::Tensor target = train_dataset->targets();
    
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(train_dataset->map(torch::data::transforms::Stack<>()), 64);
    auto batch_iter = data_loader->begin();

    torch::Tensor image_batch = batch_iter->data;
    torch::Tensor target_batch = batch_iter->target;

    torch::Tensor image_show = image_batch.index_select(0, torch::arange(0, 4));
    showBatchImage(image_show);

    torch::optim::SGD optimizer(sm->parameters(), 0.1);

    long batch_index = 0;
    for (size_t epoch = 1; epoch <= 20; epoch++)
    {
        for (auto &batch : *data_loader)
        {
        auto image_batch = batch.data;
        auto target_batch = batch.target;
        optimizer.zero_grad();
        auto y_hat = sm->forward(image_batch);
        auto loss = torch::nll_loss(y_hat, target_batch);
        loss.backward();
        optimizer.step();
        if (++batch_index % 500 == 0)
        {
            std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
            // Serialize your model periodically as a checkpoint.
            torch::save(sm, "/home/liumu/code/DLTurtle/model/net.pt");
        }
        }
    }

    std::cout << std::endl
                << "\r\nTraining finished!\r\n"
                << std::endl;