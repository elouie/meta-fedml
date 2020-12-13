from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Softmax


def create_emnist_model():
    """
        Roughly the LeCun-5 convolutional model (minus random connections).
    """
    return Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(filters=32, kernel_size=3, activation=relu, padding="same"),
        MaxPool2D(pool_size=2),
        Conv2D(filters=32, kernel_size=3, activation=relu, padding="same"),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="relu"),
        Softmax(),
    ])
