import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def scale(X_train, X_test):
    X_train_max = np.max(X_train)
    return X_train / X_train_max, X_test / X_train_max


def xavier(n_in, n_out):
    limit = np.sqrt(6) / np.sqrt(n_in + n_out)
    # Changed this as the seed function was generating wrong numbers for this stage's test
    flat_weights = np.random.uniform(-limit, limit, n_in * n_out)
    return flat_weights.reshape((n_in, n_out))


def sigmoid(x):
    # Simplified the sigmoid 
    return 1 / (1 + np.exp(-x))

# Class which computes and stores the next layer of the network
class OneLayerNeural:

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        # First the biases as the generation works for the project test
        self.biases = np.squeeze(xavier(1, n_classes))
        self.weights = xavier(n_features, n_classes)

        self.z = None
        self.a = None

    def forward(self, X):
        z = np.dot(X, self.weights) + self.biases
        a = sigmoid(z)

        self.z = z
        self.a = a

        return a.flatten().tolist()


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # Set the seed here to make the random numbers predictable
    np.random.seed(3042022)

    # Scale the data
    scaled_train, scaled_test = scale(X_train, X_test)

    # Create an instance of the neural network
    n_features = scaled_train.shape[1]
    n_classes = 10
    nn_1 = OneLayerNeural(n_features, n_classes)

    # Apply the model to the first two items of the training dataset
    first_two_images = scaled_train[0:2]
    results = nn_1.forward(first_two_images)

    # Print the result
    print(results)
