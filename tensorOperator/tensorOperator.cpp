//
// Created by liumu on 23-4-3.
//

#include "tensorOperator.h"

void tensorOperator::printTensor()
{
    std::cout << tensor << std::endl;
    // Create a tensor with 3 dimensions.
    auto tensor3D =
        torch::tensor({{{1, 2, 3}, {3, 2, 1}}, {{1, 2, 3}, {3, 2, 1}}});

    // Create a tensor with random integers.
    tensor = torch::randint(0, 10, {2, 3});

    // Create a tensor with random numbers from the standard normal distribution.
    tensor = torch::randn({2, 3});

    // Create a tensor from vector data.
    std::vector<float> v = {1, 2, 3, 4, 5, 6};
    tensor = torch::from_blob(v.data(), {1, 6}, torch::kFloat32);

    // Create a tensor with the same size as a given tensor and with random data.
    tensor = torch::rand_like(tensor3D, torch::kFloat32);

    // Add the tensor to itself.
    tensor += tensor;

    // Create a tensor from the given data.
    tensor = torch::tensor({1, 2, 3});

    // Create a tensor of size 10 and fill it with 3.
    torch::Tensor b = torch::full({10}, 3);

    // Print the tensor.
    std::cout << b << std::endl;

    // View the tensor as a 2D tensor of size 1x10.
    b = b.view({1, 2, -1});

    // Print the tensor.
    std::cout << b << std::endl;

    // Transpose the tensor.
    torch::Tensor c = b.transpose(0, 1);

    // Print the tensor.
    std::cout << c << std::endl;

    // Reshape the tensor.
    auto d = b.reshape({1, 1, -1});

    // Print the tensor.
    std::cout << d << std::endl;

    // Permute the tensor.
    auto e = b.permute({1, 0, 2});

    // Print the tensor.
    std::cout << e << std::endl;

    // Create a tensor of size 10x3x28x28.
    b = torch::rand({10, 3, 28, 28});

    // Print the size of the 0th picture.
    std::cout << b[0].sizes();

    // Print the size of the 0th picture, 0th channel.
    std::cout << b[0][0].sizes();

    // Print the size of the 0th picture, 0th channel, 0th row pixels.
    std::cout << b[0][0][0].sizes();

    // Print the size of the 0th picture, 0th channel, 0th row, 0th column pixels.
    std::cout << b[0][0][0][0].sizes() << std::endl;

    // Print the tensor.
    std::cout << tensor.reshape({1, -1}) << std::endl;

    std::cout << b.index_select(0, torch::tensor({0, 3, 3})).sizes() << std::endl; // choose 0th dimension at 0,3,3 to form a tensor of [3,3,28,28]
    std::cout << b.index_select(1, torch::tensor({0, 2})).sizes() << std::endl;    // choose 1th dimension at 0 and 2 to form a tensor of[10, 2, 28, 28]
    std::cout << b.index_select(2, torch::arange(0, 8)).sizes() << std::endl;      // choose all the pictures' first 8 rows [10, 3, 8, 28]
    std::cout << b.narrow(1, 0, 2).sizes() << std::endl;                           // choose 1th dimension, from 0, cutting out a lenth of 2, [10, 2, 28, 28]
    std::cout << b.select(3, 2).sizes() << std::endl;                              // select the second tensor of the third dimension, that is, the tensor composed of the second row of all pictures [10, 3, 28]

    c = torch::randn({3, 4});
    auto mask = torch::zeros({3, 4});
    mask[0][0] = 1;
    mask[2][2] = 1;
    std::cout << c << std::endl;
    std::cout << c.index_put_({mask.to(torch::kBool)}, c.index({mask.to(torch::kBool)}) + 1.5) << std::endl;
    std::cout << c << std::endl;
    std::cout << c.index({mask.to(torch::kBool)}) << std::endl;

    b = torch::ones({3, 4});
    c = torch::zeros({3, 4});
    std::cout << torch::cat({b, c}, 1) << std::endl;
    std::cout << torch::cat({b, c}, 0) << std::endl;
    std::cout << torch::stack({b, c}, 0) << std::endl;

    std::cout << torch::arange(0, 8) << std::endl;

    b = torch::rand({3, 4});
    c = torch::rand({3, 4});
    std::cout << b << c << b * c << b / c << b.mm(c.t());
}

void tensorOperator::setTensor(torch::Tensor &t) { this->tensor = t.clone(); }

void tensorOperator::getTensor(torch::Tensor &t) const { t = this->tensor.clone(); }

void tensorOperator::autoDiff()
{
    tensor.requires_grad_(true);
    auto y = 2 * torch::dot(tensor, tensor);
    y.backward();

    tensor.grad().zero_();
    y = tensor * tensor;
    y.sum().backward();

    tensor.grad().zero_();
    y = tensor * tensor;
    auto u = y.detach();
    auto z = u * tensor;
    z.sum().backward();

    tensor.grad().zero_();
    y.sum().backward();

    auto f = [](const torch::Tensor &x)
    {
        auto b = x * 2;
        torch::Tensor c;
        while (b.norm().item<float>() < 1000)
        {
            b = b * 2;
        }
        if (b.sum().item<float>() > 0)
        {
            c = b;
        }
        else
        {
            c = 100 * b;
        }
        return c;
    };

    auto x = torch::randn({1, 1}, torch::requires_grad());
    y = f(x);
    y.backward();
    std::cout << (x.grad() == y / x) << std::endl;
}

void tensorOperator::prob()
{
    at::Tensor out = at::empty({6}, tensor.options().dtype(at::kLong));
    tensor = torch::ones({6}) / 6;
    at::multinomial_out(out, tensor, 6, true);
    std::cout << out << std::endl;
}