# 🧠 Neural Network for Image Classification from Scratch

This project provides a hands-on implementation of a fully connected neural network from the ground up, using only **NumPy**. It's designed to build a solid, first-principles understanding of how neural networks work, from the math behind them to the code that brings them to life.

Instead of relying on high-level libraries like Keras or PyTorch, you will implement the core components yourself: feedforward propagation, backpropagation, weight initialization, and loss calculation. The network is trained on the **Fashion-MNIST** dataset, a popular alternative to the classic MNIST dataset for image classification tasks.

## ✨ Features

- **Pure NumPy Implementation**: Built entirely from scratch to demonstrate the core mechanics of neural networks.
- **Feedforward Propagation**: A function to pass data through the network and generate predictions.
- **Backpropagation Algorithm**: The core learning mechanism, implemented to adjust weights and biases based on error.
- **Xavier Weight Initialization**: A smart initialization technique to prevent gradients from vanishing or exploding.
- **Custom Math Functions**: Implementation of the Sigmoid activation function, Mean Squared Error (MSE) loss, and their derivatives.
- **Data Preprocessing**: Includes functions for scaling features and one-hot encoding labels.
- **Learning Visualization**: A helper function to plot loss and accuracy over epochs.

## 🛠️ Technologies Used

*   **Core**: Python
*   **Numerical Computing**: NumPy
*   **Data Handling**: Pandas
*   **Data Visualization**: Matplotlib
*   **Data Fetching**: Requests

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8+

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/michalglomsky/Neural-Network-from-Scratch.git
    cd Neural-Network-from-Scratch
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Usage

To run the script and see the neural network in action for a single training step:

```sh
python neural_network_from_scratch.py
```

### What the Script Does:

1.  **Downloads Data**: If the Fashion-MNIST dataset (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) is not found, the script will automatically download it from Dropbox into a `../Data` directory.
2.  **Preprocesses Data**: It loads the data, scales the pixel values to a range between 0 and 1, and one-hot encodes the labels.
3.  **Initializes Network**: It creates a single-layer neural network with Xavier-initialized weights.
4.  **Performs One Training Step**:
   *   It takes the first two images from the training set and performs a **forward pass**.
   *   It calculates the error and performs a **backpropagation** step to update the network's weights and biases.
   *   It performs a second **forward pass** with the updated weights to show that the loss has decreased.
5.  **Prints Output**: The script prints the initial and final Mean Squared Error for the two samples, demonstrating that the network has learned and improved its prediction.

## 🧠 Core Concepts Explained

*   **Forward Propagation**: This is the process of "predicting." Input data is fed into the network, multiplied by the weights, has the bias added, and is passed through an activation function (`sigmoid`). The output is the network's prediction.

*   **Backpropagation**: This is the process of "learning." The network's prediction is compared to the true label to calculate an error (loss). This error is then propagated backward through the network, and calculus (specifically, the chain rule) is used to determine how much each weight and bias contributed to the error. The weights and biases are then adjusted slightly to reduce the error.

*   **Xavier Initialization**: Randomly initializing weights is crucial, but poor initialization can lead to problems. Xavier initialization sets the initial weights based on the number of input and output neurons, which helps keep the signal in a reasonable range and makes training more stable.

## 🗺️ Roadmap

- [ ] **Full Training Loop**: Implement a complete training loop that iterates over multiple epochs and batches of data.
- [ ] **Evaluation on Test Set**: Add a function to evaluate the network's accuracy on the unseen test dataset after training.
- [ ] **Multi-Layer Network**: Extend the `OneLayerNeural` class to a `MultiLayerNeural` class to build deeper networks.
- [ ] **More Activation Functions**: Implement other popular activation functions like ReLU and Tanh and their derivatives.
- [ ] **Cross-Entropy Loss**: Implement Cross-Entropy Loss, which is better suited for classification tasks than MSE.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

Created by Michał Głomski

