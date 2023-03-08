//
// Created by jaehoon on 23. 3. 6.
//

#include <vector>
#include <iostream>
#include <torch/torch.h>

auto dataTensors = torch::randn({ 1000, 7 });
std::vector<float> tempCSVValues;
std::ifstream ifs("../data/GRFTrainingData.csv");
int cols = 7;
int rows = 1000;

class swish : torch::nn::Module
{
public:
    torch::Tensor forward(torch::Tensor x)
    {
        x = x * torch::sigmoid(x);
        return x;
    }
};

struct Net : torch::nn::Module
{
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr }, fc4{ nullptr }, fc5{ nullptr }, fc6{ nullptr }, fc7{ nullptr }, fc8{ nullptr }, fc9{ nullptr };
    swish activationSwish;

    Net(int in_dim, int out_dim)
    {
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, 32));
        fc2 = register_module("fc2", torch::nn::Linear(32, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 32));
        fc4 = register_module("fc4", torch::nn::Linear(32, 32));
        fc5 = register_module("fc5", torch::nn::Linear(32, 32));
        fc6 = register_module("fc6", torch::nn::Linear(32, 32));
        fc7 = register_module("fc7", torch::nn::Linear(32, 32));
        fc8 = register_module("fc8", torch::nn::Linear(32, 32));
        fc9 = register_module("fc9", torch::nn::Linear(32, out_dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = fc1->forward(x);
        x = fc2->forward(activationSwish.forward(x));
        x = fc3->forward(activationSwish.forward(x));
        x = fc4->forward(activationSwish.forward(x));
        x = fc5->forward(activationSwish.forward(x));
        x = fc6->forward(activationSwish.forward(x));
        x = fc7->forward(activationSwish.forward(x));
        x = fc8->forward(activationSwish.forward(x));
        x = fc9->forward(activationSwish.forward(x));

        return x;
    }
};

void loadCSVFile()
{
    float x = 0;
    char dummy;

    for (int i = 0; i < rows; ++i)
    {
        for (int i = 0; i < cols; ++i)
        {
            ifs >> x;
            tempCSVValues.push_back(x);
            if (i < (cols - 1))
            {
                ifs >> dummy;
            }
        }
    }
    std::cout << "CSV file is loaded." << std::endl;
}

void vec2tensor()
{
    auto tensorAccessor = dataTensors.accessor<float, 2>();

    for (int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols ; j++)
        {
            tensorAccessor.data()[i * cols + j] = tempCSVValues[i * cols + j];
        }
    }
    std::cout << "Values are saved in tensors." << std::endl;
}

auto net = std::make_shared<Net>(6, 1);

void doLearning()
{
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));

    int batchSize = 16;
    float loss_val;
    auto inputs = torch::randn({ batchSize, 6 });
    auto target = torch::randn({ batchSize, 1 });
    auto datasets = torch::data::datasets::TensorDataset(dataTensors);
    auto inputTensorAccessor = inputs.accessor<float, 2>();
    auto targetTensorAccessor = target.accessor<float, 2>();
    std::size_t maxEpochs = 300;

    for (std::size_t epoch = 1; epoch <= maxEpochs; epoch++)
    {
        auto dataLoader = torch::data::make_data_loader(datasets, torch::data::DataLoaderOptions().batch_size(batchSize));

        for (auto& batch : *dataLoader)
        {
            optimizer.zero_grad();

            for (int64_t i = 0; i < batch.size(); ++i)
            {
                inputTensorAccessor.data()[i * 6] = batch.data()[i].data[0].item<float>();
                inputTensorAccessor.data()[i * 6 + 1] = batch.data()[i].data[1].item<float>();
                inputTensorAccessor.data()[i * 6 + 2] = batch.data()[i].data[2].item<float>();
                inputTensorAccessor.data()[i * 6 + 3] = batch.data()[i].data[3].item<float>();
                inputTensorAccessor.data()[i * 6 + 4] = batch.data()[i].data[4].item<float>();
                inputTensorAccessor.data()[i * 6 + 5] = batch.data()[i].data[5].item<float>();
                targetTensorAccessor.data()[i] = batch.data()[i].data[6].item<float>();
            }
            auto out = net->forward(inputs);
            auto loss = torch::mse_loss(out, target);
            loss_val = loss.item<float>();
            loss.backward();
            optimizer.step();
        }
        std::cout<<"====== epochs : "<<epoch<<"======"<<std::endl << std::endl;
        std::cout << "Loss: " << loss_val << std::endl;
        std::cout<<std::endl << std::endl;
    }
}

void predict(float& predictValue, int idx)
{
    auto inputs = torch::randn({ 1, 6 });
    auto inputTensorAccessor = inputs.accessor<float, 2>();
    inputTensorAccessor.data()[0] = dataTensors.data()[idx].data()[0].item<float>();
    inputTensorAccessor.data()[1] = dataTensors.data()[idx].data()[1].item<float>();
    inputTensorAccessor.data()[2] = dataTensors.data()[idx].data()[2].item<float>();
    inputTensorAccessor.data()[3] = dataTensors.data()[idx].data()[3].item<float>();
    inputTensorAccessor.data()[4] = dataTensors.data()[idx].data()[4].item<float>();
    inputTensorAccessor.data()[5] = dataTensors.data()[idx].data()[5].item<float>();
    predictValue = net->forward(inputs).item<float>();

    std::cout << "target : " << dataTensors.data()[idx].data()[6].item<float>()<< std::endl;
    std::cout << "predict : " << predictValue << std::endl;
}

int main()
{
    loadCSVFile();
    vec2tensor();
    doLearning();
    float predicted;
    predict(predicted, 200);
}