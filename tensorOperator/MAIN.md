# using guide
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    torch::cuda::is_available(); // check if cuda is available
    // Create a tensor operator object.
    tensorOperator tensor_operator;

    // Create a tensor with random data.
    torch::Tensor tensor = torch::rand({2, 3});

    // Set the tensor in the tensor operator object.
    tensor_operator.setTensor(tensor);