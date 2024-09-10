# Sentiment Analysis using NLP

This repository contains a simple Sentiment Analysis project using Natural Language Processing (NLP) techniques and TensorFlow. The model is trained on the [Kaggle Sentiment Analysis dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset). It classifies text into **positive**, **neutral**, or **negative** sentiments.

## Features
- Preprocessing of text data (cleaning and transforming)
- Universal Sentence Encoder (USE) for embedding textual data
- Deep learning model with dense layers and dropout for regularization
- Early stopping to prevent overfitting
- Prediction of sentiment for user-inputted sentences

## Tech Stack
- Python
- TensorFlow 2.8.0
- TensorFlow Hub
- Universal Sentence Encoder (USE)
- Pandas, NumPy, Scikit-learn

## Getting Started

### 1. Install Dependencies:
Install the required libraries using pip:
```
pip install tensorflow==2.8.0 tensorflow-hub==0.12.0
```

### 2. Clone the Repository:
```
git clone https://github.com/yourusername/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 3. Download Dataset:
Place the train and test datasets in the same directory.

### 4. Run the Notebook:
Launch the notebook or script to train the model:
```
python Sentiment_Analysis.ipynb
```

## Key Components
- **Data Preprocessing**: Text cleaning and normalization.
- **Model**: Uses Universal Sentence Encoder for embedding and a custom deep neural network with multiple dense layers and dropout to avoid overfitting.
- **Evaluation**: Accuracy, precision, recall, and F1-score metrics are used for evaluating the model.
- **User Input**: The model can predict the sentiment of custom sentences entered by the user.

## Example Usage

```
Enter your sentence: I love this product!
Emotion: Positive 😊
```
