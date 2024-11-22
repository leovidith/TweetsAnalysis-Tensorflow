# Twitter tweets Analysis Project

This project implements a sentiment analysis model using TensorFlow and TensorFlow Hub. Below is the setup and dependency information to ensure compatibility and smooth functioning of the code.

---

## Dataset Visualization
<img src="https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/bar%20graph.png" width="600">
<img src="https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/density%20plot.png" width="600">
<img src="https://github.com/leovidith/TweetsAnalysis-Tensorflow/blob/main/images/pie%20chart.png" width="600">


## Project Overview
This repository includes a deep learning model for sentiment analysis. The model is built using TensorFlow and utilizes pre-trained embeddings from TensorFlow Hub.

## Dependencies

To ensure proper functionality, please use the specific versions of the libraries below:

- **Python**: 3.6, 3.7, or 3.8 (recommended: Python 3.7)
- **TensorFlow**: 2.8.0
- **TensorFlow Hub**: 0.12.0
- **Numpy**: 1.19.5
- **Protobuf**: 3.19.4
- **TensorBoard**: 2.8.0
- **h5py**: 3.1.0
- **six**: 1.15.0
- **grpcio**: 1.43.0
- **wheel**: 0.36.2

## Installation

To set up the environment, follow these installation instructions:

1. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

2. **Install the dependencies**:
    ```bash
    pip install tensorflow==2.8.0
    pip install tensorflow-hub==0.12.0
    pip install numpy==1.19.5
    pip install protobuf==3.19.4
    pip install tensorboard==2.8.0
    pip install h5py==3.1.0
    pip install six==1.15.0
    pip install grpcio==1.43.0
    pip install wheel==0.36.2
    ```

## Usage

Run the Python scripts or Jupyter notebooks provided in the repository to train and test the sentiment analysis model.

```bash
python script_name.py  # Replace with the actual script name
```

## Model Training and Saving

This project uses a pre-trained Universal Sentence Encoder from TensorFlow Hub and fine-tunes it using a neural network built with Keras.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
