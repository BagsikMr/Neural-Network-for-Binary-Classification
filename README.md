# Neural Network for Binary Classification

This repository contains a basic implementation of a neural network using TensorFlow to solve a binary classification problem. The neural network is trained to classify data points based on their coordinates falling within a specified radius.

## Requirements
- Python (>=3.6)
- NumPy
- TensorFlow (>=2.0)

## Usage

1. Clone the repository:

```shell
git clone github.com/BagsikMr/Neural-Network-for-Binary-Classification.git
```

2. Navigate to the project directory:

```shell
cd neural-network-binary-classification
```

3. Install the required dependecies

```shell
pip install numpy
pip install tensorflow
```

4. Run the code
```shell
python main.py
```

## Code Explanation
- The `generate_data` function generates random data points and their corresponding labels based on whether they fall within a specified radius or not.
- The neural network model is defined using the `Sequential` class from TensorFlow's Keras API. It consists of three fully connected (dense) layers with ReLU activation functions, and the last layer uses a sigmoid activation function for binary classification.
- The model is trained using the Adam optimizer and binary cross-entropy loss function.
- After training, the model is tested on a separate set of data points to evaluate its accuracy.

## Customization
- You can modify the `num_samples` and `radius` variables in the code to control the number of training samples and the radius for generating the data.
- The structure and parameters of the neural network model can be adjusted as per your requirements.
