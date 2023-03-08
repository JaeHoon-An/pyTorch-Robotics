#include "mainwindow.h"
#include "ui_mainwindow.h"

/*
 * TODO: Implement file select mode when loading datasets.
 * TODO: Implement standard deviation of loss in each epoch.
 */
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , mBatchSize(DEFAULT_BATCH_SIZE)
    , mDatasetColum(7)
    , mDatasetRow(1000)
    , mInputDimension(6)
    , mOutputDimension(1)
    , mEpochs(DEFAULT_EPOCHS)
    , mCurrentEpoch(0)
    , mEstimateIndex(0)
    , mLearningRate(DEFAULT_LEARNING_RATE)
    , mMiniBatchLoss(0.0)
    , mLossMean(0.0)
    , mMSE(0.0)
    , mEstimatedValue(0.0)
    , mGroundTruthValue(0.0)
{
    ui->setupUi(this);
    initializeDisplay();
    initializeGraph();

    net = std::make_shared<Net>(mInputDimension, mOutputDimension);
    mDataTensors = torch::randn({ mDatasetRow, mDatasetColum });
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::initializeDisplay()
{
    ui->LE_DATASET_LOAD_PATH->setText("Datasets.csv");
    ui->LE_PARAMS_BATCH_SIZE->setText(QString().sprintf("%d", mBatchSize));
    ui->LE_PARAMS_EPOCHS->setText(QString().sprintf("%d", mEpochs));
    ui->LE_PARAMS_LAERNING_RATE->setText(QString().sprintf("%f", mLearningRate));
    ui->LE_MODEL_SAVE_PATH->setText("Model.pt");
    ui->LE_MODEL_LOAD_PATH->setText("Model.pt");
}

void MainWindow::initializeGraph()
{
    QPen myPen, dotPen;
    myPen.setWidthF(1.5);
    myPen.setColor(Qt::blue);
    dotPen.setStyle(Qt::DotLine);
    dotPen.setWidth(20);
    dotPen.setWidthF(2);
    dotPen.setColor(Qt::gray);

    QSharedPointer<QCPAxisTickerTime> timeTicker(new QCPAxisTickerTime);
    timeTicker->setTimeFormat("%m:%s");

    ui->QCP_LEARNING->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP :: iSelectAxes | QCP::iSelectLegend | QCP::iSelectPlottables);
    ui->QCP_ESTIMATE->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP :: iSelectAxes | QCP::iSelectLegend | QCP::iSelectPlottables);

    ui->QCP_LEARNING->legend->setVisible(true);
    ui->QCP_LEARNING->legend->setFont(QFont("Helvetica", 9));
    ui->QCP_LEARNING->addGraph();
    ui->QCP_LEARNING->graph(0)->setPen(myPen);
    ui->QCP_LEARNING->graph(0)->setName("Loss - Mean");

    ui->QCP_ESTIMATE->legend->setVisible(true);
    ui->QCP_ESTIMATE->legend->setFont(QFont("Helvetica", 9));
    ui->QCP_ESTIMATE->addGraph();
    ui->QCP_ESTIMATE->graph(0)->setPen(myPen);
    ui->QCP_ESTIMATE->graph(0)->setName("Estimated value");
    ui->QCP_ESTIMATE->addGraph();
    ui->QCP_ESTIMATE->graph(1)->setPen(dotPen);
    ui->QCP_ESTIMATE->graph(1)->setName("Ground-true");

    ui->QCP_LEARNING->xAxis->setRange(0, mEpochs);
    ui->QCP_ESTIMATE->xAxis->setRange(0, mDatasetRow);
    ui->QCP_LEARNING->yAxis->setRange(0, 10);
    ui->QCP_ESTIMATE->yAxis->setRange(0, 30);
}

void MainWindow::updateDisplay()
{
    ui->LE_LEARNING_MINI_BATCH_LOSS->setText(QString().sprintf("%f", mMiniBatchLoss));
    ui->LE_ESTIMATE_MSE->setText(QString().sprintf("%f", mMSE));
}

void MainWindow::updateGraph()
{
    ui->QCP_LEARNING->graph(0)->addData(mCurrentEpoch, mMiniBatchLoss);
    ui->QCP_ESTIMATE->graph(0)->addData(mEstimateIndex, mEstimatedValue);
    ui->QCP_ESTIMATE->graph(1)->addData(mEstimateIndex, mGroundTruthValue);

    ui->QCP_LEARNING->xAxis->setRange(0, mEpochs);
    ui->QCP_ESTIMATE->xAxis->setRange(0, mDatasetRow);

    ui->QCP_LEARNING->replot();
    ui->QCP_ESTIMATE->replot();
}

void MainWindow::on_BT_DATASET_LOAD_clicked()
{
    std::string dataPath;
    dataPath.append(DATA_DIR);
    dataPath.append(ui->LE_DATASET_LOAD_PATH->text().toStdString());
    loadData(dataPath);

}

void MainWindow::loadData(std::string& dataPath)
{
    std::ifstream ifs(dataPath);
    std::vector<float> datasets;

    try
    {
        if (!ifs.is_open())
        {
            throw dataPath;
        }
        float x = 0;
        char dummy;

        for (int i = 0; i < mDatasetRow; ++i)
        {
            for (int i = 0; i < mDatasetColum; ++i)
            {
                ifs >> x;
                datasets.push_back(x);
                if (i < (mDatasetColum - 1))
                {
                    ifs >> dummy;
                }
            }
        }
        mVectorDatasets = datasets;
        std::cout << "[DATA LOADER] Datasets are loaded." << std::endl;
        ifs.close();
        vec2tensor();
        std::cout << "[DATA LOADER] Datasets are transferred to Tensors." << std::endl << std::endl;
    }
    catch (std::string path)
    {
        perror("[DATA LOADER] Datasets are not loaded.");
        std::cout << "[DATA LOADER] Given path : " << path << std::endl << std::endl;
    }
}

void MainWindow::vec2tensor()
{
    auto tensorAccessor = mDataTensors.accessor<float, 2>();

    for (int i = 0; i < mDatasetRow; i++)
    {
        for(int j = 0; j < mDatasetColum ; j++)
        {
            tensorAccessor.data()[i * mDatasetColum + j] = mVectorDatasets[i * mDatasetColum + j];
        }
    }
}

void MainWindow::on_BT_LEARNING_clicked()
{
    ui->QCP_LEARNING->graph(0)->data()->clear();

    mBatchSize = ui->LE_PARAMS_BATCH_SIZE->text().toInt();
    mLearningRate = ui->LE_PARAMS_LAERNING_RATE->text().toFloat();
    mEpochs = ui->LE_PARAMS_EPOCHS->text().toInt();
    std::cout << "[PARAMETERS] Batch size\t\t : " << mBatchSize << std::endl;
    std::cout << "[PARAMETERS] Learning rate\t : " << mLearningRate << std::endl;
    std::cout << "[PARAMETERS] Epochs\t\t\t : " << mEpochs << std::endl;

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(mLearningRate));

    auto inputs = torch::randn({ mBatchSize, mInputDimension });
    auto target = torch::randn({ mBatchSize, mOutputDimension });
    auto datasets = torch::data::datasets::TensorDataset(mDataTensors);
    auto inputTensorAccessor = inputs.accessor<float, 2>();
    auto targetTensorAccessor = target.accessor<float, 2>();

    for (std::size_t epoch = 1; epoch <= mEpochs; epoch++)
    {
        int iteration = 0;
        mLossMean = 0;
        auto dataLoader = torch::data::make_data_loader(datasets, torch::data::DataLoaderOptions().batch_size(mBatchSize));
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
            mMiniBatchLoss = loss.item<float>();
            loss.backward();
            optimizer.step();
            mLossMean += mMiniBatchLoss;
            iteration++;
            updateDisplay();
        }
        mLossMean /= iteration;
        mCurrentEpoch = epoch;
        std::cout << "[LEARNING] Epoch : " << mCurrentEpoch << "\tAverage Loss : " << mLossMean << std::endl;
        updateGraph();
    }
    std::cout << std::endl;
}

void MainWindow::on_BT_MODEL_SAVE_clicked()
{
    std::cout << "Model save is clicked." << std::endl;
    std::string modelPath;
    modelPath.append(MODEL_DIR);
    modelPath.append(ui->LE_MODEL_SAVE_PATH->text().toStdString());
    torch::save(net, modelPath);
    std::cout << "[MODEL MANAGER] : " << ui->LE_MODEL_SAVE_PATH->text().toStdString() << " is saved." << std::endl << std::endl;
}

void MainWindow::on_BT_MODEL_LOAD_clicked()
{
    std::string modelPath;
    modelPath.append(MODEL_DIR);
    modelPath.append(ui->LE_MODEL_LOAD_PATH->text().toStdString());
    std::ifstream ifs(modelPath);
    try
    {
        if(!ifs.is_open())
        {
            throw modelPath;
        }
        torch::load(net, modelPath);
        std::cout << "[MODEL MANAGER] : " << ui->LE_MODEL_LOAD_PATH->text().toStdString() << " is loaded." << std::endl << std::endl;
    }
    catch(std::string path)
    {
        perror("[MODEL MANAGER] Model is not loaded.");
        std::cout << "[MODEL MANAGER] Given path : " << path << std::endl << std::endl;
    }
    ifs.close();
}

void MainWindow::on_BT_MODEL_ESTIMATE_clicked()
{
    auto inputs = torch::randn({ 1, mInputDimension });
    auto inputTensorAccessor = inputs.accessor<float, 2>();
    float squaredError = 0;
    float sumOfSquaredError = 0;
    ui->QCP_ESTIMATE->graph(0)->data()->clear();
    ui->QCP_ESTIMATE->graph(1)->data()->clear();
    for(int i = 0 ; i < mDatasetRow ; i++)
    {
        inputTensorAccessor.data()[0] = mDataTensors.data()[i].data()[0].item<float>();
        inputTensorAccessor.data()[1] = mDataTensors.data()[i].data()[1].item<float>();
        inputTensorAccessor.data()[2] = mDataTensors.data()[i].data()[2].item<float>();
        inputTensorAccessor.data()[3] = mDataTensors.data()[i].data()[3].item<float>();
        inputTensorAccessor.data()[4] = mDataTensors.data()[i].data()[4].item<float>();
        inputTensorAccessor.data()[5] = mDataTensors.data()[i].data()[5].item<float>();
        mGroundTruthValue = mDataTensors.data()[i].data()[6].item<float>();
        mEstimateIndex = i;
        mEstimatedValue = net->forward(inputs).item<float>();
        squaredError =  pow((mGroundTruthValue - mEstimatedValue), 2.0);
        sumOfSquaredError += squaredError;
        mMSE = pow(squaredError, 0.5) / (i + 1);
        updateDisplay();
        updateGraph();
        std::cout << "[ESTIMATION] Index : " << mEstimateIndex << "\tSquared Error : " << squaredError << "\tEstimated Value : " << mEstimatedValue << "\tGround-true Value : " << mGroundTruthValue << std::endl;
    }
    std::cout << std::endl;
}
