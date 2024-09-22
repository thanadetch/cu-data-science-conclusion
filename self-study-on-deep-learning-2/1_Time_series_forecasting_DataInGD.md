## How to use PyTorch LSTMs/GRUs for time series regression

### 1. **Describing Models, Hardware Requirements, Data Statistics, Learning Curves, Metrics, and Finetuning Techniques**

#### **1.1 Model Input and Output**
- **Input**: In time series forecasting with LSTM/GRU models, the input is typically a sequence of historical data points. For stock forecasting, this might include features such as:
    - Date
    - Opening Price
    - High Price
    - Low Price
    - Closing Price
    - Volume

  The input format for LSTM models is usually:  `(batch_size, sequence_length, input_features)`


- **Output**: The output is a prediction of the future value(s), for instance, the stock price at a given time horizon.
    - The output shape is typically: `(batch_size, prediction_length)` where `prediction_length` corresponds to the number of time steps you're forecasting.

#### **1.2 Hardware Requirements**
- **Basic Requirements**:
    - **CPU**: Time series models can run on CPUs, but training will be slower.
    - **GPU**: For faster training, especially with large datasets or deep networks, using a GPU (such as NVIDIA) is highly recommended. CUDA-capable GPUs with PyTorch can speed up training significantly.

#### **1.3 Data Statistics**
- **Statistical Analysis**:
    - **Mean**: The average of the stock prices.
    - **Standard Deviation**: The volatility or how much stock prices deviate from the mean.
    - **Min/Max Values**: Range of stock prices over the dataset.
    - **Data Split**: Typically, time series data is split into training, validation, and test sets.
        - **Training set**: Often around 70% of the data.
        - **Validation set**: Used for hyperparameter tuning and model selection.
        - **Test set**: Final model evaluation.

#### **1.4 Learning Curve**
- The learning curve demonstrates how the model's performance evolves over time during training.
    - **Training Loss**: Typically, Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) is tracked on the training data.
    - **Validation Loss**: These same metrics are tracked on the validation dataset to detect overfitting or underfitting.

A plot of training and validation loss can help diagnose model performance and finetuning needs.

#### **1.5 Metrics**
- **Training Metrics**:
    - **MSE (Mean Squared Error)**: A measure of the average squared difference between the actual values and the predicted values.
    - **RMSE (Root Mean Squared Error)**: Square root of MSE, giving error in the same units as the predicted values.
    - **MAE (Mean Absolute Error)**: The average of the absolute errors between predictions and true values.
- **Validation Metrics**: Similar to training metrics, validation metrics track the performance on unseen validation data. If the validation error remains high while training error decreases, the model may be overfitting.

#### **1.6 Demo the Result**
- A demonstration might include:
    - Plotting the true stock prices against the predicted stock prices over time.
    - Showing the error metrics (MSE, RMSE) for the predictions on test data.

#### **1.7 Fine-Tuning Techniques**
- **Learning Rate Tuning**: Adjusting the step size for optimization algorithms like Adam or SGD can help achieve convergence faster or avoid overshooting the minimum.
- **Early Stopping**: Stop training if the validation loss does not improve for a number of epochs.
- **Batch Size Tuning**: Smaller batch sizes can offer more accurate updates but at the cost of longer training times.
- **Sequence Length Tuning**: Varying how many past time steps are used in each input sequence can impact model performance.
- **Regularization**: Techniques like dropout or L2 regularization can help prevent overfitting.

---

### 2. **Cheat Sheet: Key Functions and Their Features**

| **Function**            | **Key Features**                                                                 | **Input**                                                                            | **Output**                                                 |
|-------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------|
| **LSTM/GRU Layer**       | Recurrent layer for capturing sequence data. Includes hidden states and cell states (LSTM only). | Input shape: `(batch_size, sequence_length, input_features)`                         | Output shape: `(batch_size, hidden_size)`                   |
| **Linear Layer**         | Fully connected layer, used for final prediction.                                | Input: Output of LSTM/GRU layer                                                      | Output: Final prediction value `(batch_size, output_size)`  |
| **Adam Optimizer**       | Gradient-based optimizer, frequently used with learning rates.                   | Input: Model parameters, learning rate                                               | Output: Updated parameters after each step                  |
| **Loss Function (MSE)**  | Computes the mean squared error between predicted and true values.               | Input: Predicted values, true values                                                 | Output: Scalar loss value                                   |
| **train() function**     | Puts model into training mode (activates dropout, etc.).                         | Input: None                                                                          | Output: Model in training mode                              |
| **eval() function**      | Puts model into evaluation mode (disables dropout, etc.).                        | Input: None                                                                          | Output: Model in evaluation mode                            |
| **scheduler.step()**     | Adjusts the learning rate dynamically based on performance.                      | Input: Current validation loss or step count                                         | Output: Adjusted learning rate                              |
| **DataLoader**           | Efficiently batches and shuffles data for training.                              | Input: Dataset, batch size, shuffle flag                                             | Output: Iterator for batches of data                        |
| **plot_loss_curve()**    | Visualizes the training and validation loss over time.                           | Input: Training loss, validation loss                                                | Output: Learning curve plot                                 |
| **torch.save()**         | Saves the trained model to a file.                                               | Input: Model parameters, file path                                                   | Output: Saved model                                         |
| **torch.load()**         | Loads a trained model from a file.                                               | Input: File path                                                                     | Output: Loaded model                                        |


reference:  
https://colab.research.google.com/github/pvateekul/2110531_DSDE_2024s1/blob/main/code/Week04_DL/5_Time_series_forecasting_DataInGD.ipynb
