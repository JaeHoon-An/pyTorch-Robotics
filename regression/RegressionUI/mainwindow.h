#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <torch/torch.h> // "torch.h" should be included in first.
#include <QMainWindow>
#include <vector>
#include <iostream>

constexpr int DEFAULT_BATCH_SIZE = 16;
constexpr int DEFAULT_EPOCHS = 100;
constexpr float DEFAULT_LEARNING_RATE = 0.0001;


QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void initializeDisplay();
    void initializeGraph();
    void updateDisplay();
    void updateGraph();
    void loadData(std::string& dataPath);
    void vec2tensor();
    void on_BT_MODEL_SAVE_clicked();
    void on_BT_MODEL_LOAD_clicked();
    void on_BT_DATASET_LOAD_clicked();
    void on_BT_LEARNING_clicked();
    void on_BT_MODEL_ESTIMATE_clicked();

private:
    struct swish : torch::nn::Module
    {
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

    Ui::MainWindow* ui;
    torch::Tensor mDataTensors;
    std::vector<float> mVectorDatasets;
    int mBatchSize;
    int mDatasetColum;
    int mDatasetRow;
    int mInputDimension;
    int mOutputDimension;
    int mEpochs;
    int mCurrentEpoch;
    int mEstimateIndex;
    float mLearningRate;
    float mMiniBatchLoss;
    float mLossMean;
    float mMSE;
    float mEstimatedValue;
    float mGroundTruthValue;
    std::shared_ptr<Net> net;
};

#endif // MAINWINDOW_H
