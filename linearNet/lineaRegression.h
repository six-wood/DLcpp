#if !defined(lineaRegression_h)
#define lin
#include <torch/torch.h>
#include <random>
#include <boost/coroutine2/all.hpp>

typedef boost::coroutines2::coroutine<int> coro_t;

class lineaRegression
{
public:
    lineaRegression() = default;
    lineaRegression(torch::Tensor &w, torch::Tensor &b, int num_examples)
        : w_true(w), b_true(b), num_examples(num_examples)
    {
        std::cout << this->w_true << std::endl;
        std::cout << this->b_true << std::endl;
        std::cout << this->num_examples << std::endl;
    }
    ~lineaRegression() = default;

    void generateData();

    void dataIter(int batch_size, torch::Tensor &features, torch::Tensor &labels, coro_t::push_type &yield);

    inline torch::Tensor linearRegression(torch::Tensor &X, torch::Tensor &w, torch::Tensor &b)
    {
        return torch::mm(X, w.view({w.size(0), 1})) + b;
    }

    inline torch::Tensor squared_loss(torch::Tensor &y_hat, torch::Tensor &y)
    {
        return (y_hat - y.view(y_hat.sizes())).pow(2) / 2;
    }

    inline void sgd(std::vector<torch::Tensor> &params, float lr, int batch_size)
    {
        for (auto &param : params)
        {
            param.data().sub_(lr * param.grad() / batch_size);
            param.grad().data().zero_();
        }
    }

    inline void getW(torch::Tensor &w)
    {
        w = this->w_true;
    }

    inline void getB(torch::Tensor &b)
    {
        b = this->b_true;
    }

    inline void getFeature(torch::Tensor &feature)
    {
        feature = this->feature;
    }

    inline void getLabels(torch::Tensor &labels)
    {
        labels = this->labels;
    }

private:
    torch::Tensor w_true;
    torch::Tensor b_true;

    int num_examples;
    torch::Tensor feature;
    torch::Tensor labels;
};

class LineaRegressionDataset : public torch::data::Dataset<LineaRegressionDataset>
{
public:
    LineaRegressionDataset(torch::Tensor &features, torch::Tensor &labels)
        : features(features), labels(labels) {}

    torch::data::Example<> get(size_t index) override
    {
        return {features[index], labels[index]};
    }

    torch::optional<size_t> size() const override
    {
        return features.size(0);
    }

private:
    torch::Tensor features;
    torch::Tensor labels;
};

class LineaRegressionModule : public torch::nn::Module
{
public:
    LineaRegressionModule(int num_inputs = 2, int num_outputs = 1)
        : linear(register_module("linear", torch::nn::Linear(num_inputs, num_outputs))) {}

    torch::Tensor forward(torch::Tensor x)
    {
        return linear(x);
    }

private:
    torch::nn::Linear linear;
};

#endif // lin
