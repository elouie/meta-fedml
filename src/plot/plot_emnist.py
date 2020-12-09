import tensorflow_federated as tff
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.color_palette("colorblind", 10)
plt.rcParams["figure.dpi"] = 300


def emnist_digit_counts_boxplot():
    """
        Generates a box plot of the variation in distribution of digits
        across clients.
    """
    emnist_train, _ = tff.simulation.datasets.emnist.load_data()
    # Number of examples per layer for a sample of clients
    hist = np.zeros((len(emnist_train.client_ids), 10))
    for i in range(len(emnist_train.client_ids)):
        client_dataset = emnist_train.create_tf_dataset_for_client(
            emnist_train.client_ids[i])
        for example in client_dataset:
            label = example["label"].numpy()
            hist[i, label] += 1
    df = pd.DataFrame(hist, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="Digits", y="Counts", ax=ax, data=pd.melt(df, var_name="Digits", value_name="Counts"))
    plt.savefig("images/emnist-boxplot.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xlabel("Digits")
    sns.barplot(x=df.columns, y=df.iloc[0], ax=ax)
    plt.ylabel("Counts")
    plt.savefig("images/emnist-columns.png")
