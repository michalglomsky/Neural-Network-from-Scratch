import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


# scroll to the bottom to start coding your solution


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
    X_test_max = np.max(X_test)

    return X_train / X_train_max, X_test / X_test_max

def xavier(n_in, n_out):
    # The bound of the distribution
    limit = (np.sqrt(6) / np.sqrt(n_in + n_out))
    # Create a 2D matrix and fill each row within the loop.
    weights = np.random.uniform(-limit, limit, (n_out, n_in))

    return weights

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

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

    # Print the example outputs

    scaled = scale(X_train, X_test)
    scaled_train = scaled[0].tolist()
    scaled_test = scaled[1].tolist()
    print([scaled_train[2][778], scaled_test[0][774]], (xavier(2,3)).flatten().tolist(), sigmoid([-1,0,1,2]).flatten().tolist())