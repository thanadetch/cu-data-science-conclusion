## Fine-Tuning Vision Transformers for Image Classification

### 1. **Model Input and Output, Hardware Requirements, Learning Curves, Metrics, Demo Results, and Fine-Tuning Techniques**

#### **1.1 Model Input and Output**
- **Input**:
    - Image classification models in Hugging Face typically expect images as input, preprocessed to match the model's requirements.
    - The input can be a single image or a batch of images. Each image needs to be resized to a specific size (e.g., 224x224 pixels for models like ViT or ResNet).
    - The images are normalized according to the specific model’s requirements, using mean and standard deviation values.
    - Input shape for models like Vision Transformer (ViT) or ResNet: `(batch_size, channels, height, width)`  
      Example: A batch of RGB images of size 224x224 would have the shape `(batch_size, 3, 224, 224)`.

- **Output**:
    - The output of the model is a probability distribution (or logits) over the number of classes in the dataset.
    - For instance, in a 1000-class classification task (like ImageNet), the output will have a shape of `(batch_size, 1000)`, representing the likelihood of the image belonging to each of the 1000 classes.
    - The highest probability represents the predicted class.

#### **1.2 Hardware Requirements**
- **CPU**: Hugging Face models can be trained or fine-tuned on a CPU, but this will be slow, especially for deep models.
- **GPU**: A GPU is highly recommended for faster training and inference. CUDA-capable GPUs (such as those from NVIDIA) drastically reduce training time.
- **RAM**: Sufficient RAM (16 GB or higher) is helpful, especially when handling large datasets.
- **Disk Space**: Pretrained models from Hugging Face can be several hundred MBs in size. Ensure you have enough disk space for models and datasets.

#### **1.3 Data Statistics**
- **Basic Preprocessing**:
    - Data normalization is critical. For ImageNet-based models, standard normalization involves:
      ```python
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      ```
    - **Augmentation**: Random transformations (rotations, flips, crops) are applied to images during training to avoid overfitting and increase generalization.
- **Data Split**:
    - Training Set: 70-80% of the data.
    - Validation Set: 10-20% of the data for hyperparameter tuning.
    - Test Set: 10-15% of the data for final model evaluation.

#### **1.4 Learning Curves**
- **Training Loss**: This typically decreases over time as the model learns from the training data. The loss function used is often cross-entropy loss.
- **Validation Loss**: This helps monitor how well the model generalizes to unseen data. If the validation loss starts increasing while the training loss keeps decreasing, overfitting is likely happening.
- **Accuracy**: Tracking accuracy on both the training and validation sets is important to ensure that the model is learning effectively.

A common practice is to visualize the learning curves (training vs. validation loss) to track performance across epochs.

#### **1.5 Metrics**
- **Cross-Entropy Loss**: Used for classification tasks. It calculates the difference between predicted probabilities and actual labels.
- **Accuracy**: The percentage of correctly classified samples.
- **Precision, Recall, F1-Score**: These are useful when dealing with imbalanced datasets.
- **Confusion Matrix**: A graphical representation showing how many instances were predicted correctly/incorrectly for each class.

#### **1.6 Demo the Results**
- You can visualize the results by plotting a confusion matrix or showing sample images with their predicted labels.
- **Example**:
  ```python
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt
  
  # Assuming y_true are the true labels and y_pred are the model's predictions
  cm = confusion_matrix(y_true, y_pred)
  plt.imshow(cm, cmap='Blues')
  plt.colorbar()
  plt.show()
  ```

#### **1.7 Fine-Tuning Techniques**
- **Transfer Learning**: Start with a pretrained model (e.g., `google/vit-base-patch16-224` or `resnet50`) and fine-tune it on your dataset.
- **Learning Rate Scheduling**: Use schedulers like `ReduceLROnPlateau` to adjust the learning rate when the validation loss stops improving.
- **Early Stopping**: Stop training if the validation loss does not improve after several epochs, which helps prevent overfitting.
- **Batch Size and Epochs**: Tuning these parameters can significantly impact training speed and model performance.

---

### 2. **Cheat Sheet: Key Functions and Their Features**

| **Function**             | **Key Features**                                                                 | **Input**                                                                            | **Output**                                                 |
|--------------------------|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------|
| **from_pretrained()**     | Loads a pretrained Hugging Face model (like ViT, ResNet).                         | Model name (e.g., `google/vit-base-patch16-224`)                                     | Pretrained model ready for fine-tuning or inference         |
| **Trainer**               | Handles training and evaluation of the model.                                    | Model, optimizer, dataset, metrics                                                   | Trained model, evaluation metrics                           |
| **TrainingArguments**     | Defines training configurations (batch size, learning rate, etc.).                | Batch size, epochs, learning rate, etc.                                              | Configured arguments for training                           |
| **Dataset.map()**         | Apply preprocessing/augmentation functions to the dataset.                        | Preprocessing function, dataset                                                      | Transformed dataset                                         |
| **compute_metrics()**     | Define custom metrics like accuracy, F1-score, etc.                               | Predictions, labels                                                                  | Computed metrics like accuracy, F1-score                    |
| **torch.optim.AdamW()**   | Optimizer for adjusting model weights based on gradients.                         | Model parameters, learning rate                                                      | Optimizer                                                   |
| **scheduler.step()**      | Adjust learning rate dynamically based on validation performance.                 | Current validation loss or epoch                                                     | Updated learning rate                                       |
| **model.eval()**          | Switches the model to evaluation mode (disables dropout, etc.).                   | None                                                                                 | Model in evaluation mode                                    |
| **model.train()**         | Switches the model to training mode (enables dropout, etc.).                     | None                                                                                 | Model in training mode                                      |
| **Trainer.train()**       | Executes the training loop.                                                       | None                                                                                 | Trained model                                               |
| **Trainer.evaluate()**    | Evaluates the model on the validation set.                                        | None                                                                                 | Evaluation metrics like accuracy                            |
| **AutoTokenizer.from_pretrained()** | Loads the appropriate tokenizer for Hugging Face models.                | Model name (e.g., `bert-base-uncased`)                                               | Tokenizer                                                   |
| **torch.save()**          | Saves the trained model to a file.                                                | Model parameters, file path                                                          | Saved model                                                 |
| **torch.load()**          | Loads the model from a saved file.                                                | File path                                                                            | Loaded model ready for inference or further training         |


reference:  
https://colab.research.google.com/github/pvateekul/2110531_DSDE_2024s1/blob/main/code/Week05_AdvancedML/1_Huggingface_image_classification.ipynb
