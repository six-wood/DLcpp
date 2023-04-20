#include "lineaRegression.h"

void lineaRegression::generateData()
{
    this->feature = torch::normal(0, 1, {this->num_examples, this->w_true.size(0)});
    this->labels = torch::mm(feature, this->w_true.view({this->w_true.size(0), 1})) + this->b_true;
    this->labels += torch::normal(0, 0.1, labels.sizes());
}

void lineaRegression::dataIter(int batch_size, torch::Tensor &features, torch::Tensor &labels, coro_t::push_type &yield)
{
    int num_examples = this->num_examples;
    std::vector<int> indices(num_examples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    torch::Tensor indices_t = torch::from_blob(indices.data(), {num_examples}, torch::kInt32);

    int j;
    torch::Tensor index;
    for (int i = 0; i < num_examples; i += batch_size)
    {
        j = std::min(i + batch_size, num_examples);
        index = indices_t.slice(0, i, j, 1);
        features = this->feature.index_select(0, index);
        labels = this->labels.index_select(0, index);
        yield(1);
    }
}