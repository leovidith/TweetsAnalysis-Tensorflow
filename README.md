# Twitter Tweets Analysis Project

This project implements a sentiment analysis model using TensorFlow and TensorFlow Hub.

## Dataset Visualization

The following visualizations provide insights into the dataset:

- **Bar Graph**:
  ![Bar Graph](https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/bar%20graph.png)

- **Density Plot**:
  ![Density Plot](https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/density%20plot.png)

- **Pie Chart**:
  ![Pie Chart](https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/pie%20chart.png)


## Project Overview

This repository includes a deep learning model for sentiment analysis, designed to classify tweets as positive, negative, or neutral. The model is built using **TensorFlow** and utilizes pre-trained embeddings from **TensorFlow Hub**. The process includes loading the dataset, preprocessing the text, training the sentiment analysis model, and evaluating its performance.


## Features

- **Dataset**: Twitter dataset containing labeled tweets for sentiment analysis.
- **Model**: TensorFlow-based sentiment analysis model, utilizing pre-trained embeddings from TensorFlow Hub.
- **Task**: Multi-class sentiment classification (positive, negative, neutral).
- **Framework**: TensorFlow 2 and `tf.keras`.


## Results

The following visualizations provide insights into the dataset:

- **Bar Graph**: Shows the distribution of sentiment labels across the dataset.
- **Density Plot**: Displays the distribution of sentiment scores across different classes.
- **Pie Chart**: Visualizes the percentage of positive, negative, and neutral tweets.


## Sprint Features

### **Sprint 1: Data Loading and Preprocessing**
- Load and clean the Twitter dataset, removing any irrelevant text (e.g., URLs, special characters).
- Tokenize and pad sequences for model input.
- **Deliverable**: Cleaned and preprocessed dataset.

### **Sprint 2: Model Architecture**
- Implement a neural network model using TensorFlow, utilizing a pre-trained Universal Sentence Encoder for feature extraction.
- Build a simple feed-forward neural network for sentiment classification.
- **Deliverable**: Model architecture implemented and ready for training.

### **Sprint 3: Training the Model**
- Train the sentiment analysis model with a training dataset.
- Use early stopping to prevent overfitting and monitor validation loss.
- **Deliverable**: Trained model with performance metrics.

### **Sprint 4: Model Evaluation and Optimization**
- Evaluate the model using accuracy, precision, recall, and F1 score.
- Fine-tune the model based on evaluation results for optimal performance.
- **Deliverable**: Optimized model with improved evaluation metrics.


## Conclusion

The Twitter Sentiment Analysis project demonstrates how transfer learning with pre-trained models like the Universal Sentence Encoder can be leveraged to quickly build a high-performing sentiment analysis model. The model successfully classifies tweets into positive, negative, and neutral categories. Future improvements may include experimenting with different neural network architectures and fine-tuning hyperparameters for better accuracy.

---

Let me know if you'd like further adjustments!
