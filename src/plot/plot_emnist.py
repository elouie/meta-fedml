import tensorflow_federated as tff
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
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


def emnist_fedavg_digit_counts_boxplot():
    """
        Generates a box plot of the variation in distribution of digits
        across clients.
    """
    client_ids_split = np.load("src/data/test_split.npy", allow_pickle=True)
    emnist_train, _ = tff.simulation.datasets.emnist.load_data()
    train_datasets = []
    for client_ids in client_ids_split:
        train_dataset = emnist_train.create_tf_dataset_for_client(client_ids[0])
        for i in range(1, len(client_ids)):
            train_dataset = train_dataset.concatenate(emnist_train.create_tf_dataset_for_client(client_ids[i]))
        train_datasets.append(train_dataset)
    # Number of examples per layer for a sample of clients
    hist = np.zeros((len(train_datasets), 10))
    for i, client_dataset in enumerate(train_datasets):
        for example in client_dataset:
            label = example["label"].numpy()
            hist[i, label] += 1
    df = pd.DataFrame(hist, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="Digits", y="Counts", ax=ax, data=pd.melt(df, var_name="Digits", value_name="Counts"))
    plt.savefig("images/emnist-split-boxplot.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xlabel("Digits")
    sns.barplot(x=df.columns, y=df.iloc[0], ax=ax)
    plt.ylabel("Counts")
    plt.savefig("images/emnist-split-columns.png")


def emnist_experiment_validation_accuracy_result_plot():
    # Mislabeled file. Should be
    global_sgd = pd.read_csv("src/experiments/history/global_history_ep10.csv")
    fedavg_1 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp1.csv")
    fedavg_2 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp2.csv")
    fedavg_5 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp5.csv")
    fedavg_10 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp10.csv")
    fig, ax = plt.subplots()
    N = 10
    colors = np.linspace(0, 1, N)
    cmap = ListedColormap(sns.color_palette("colorblind", N))
    ax.plot(list(global_sgd.index), global_sgd["val_accuracy"], '-', label='Global')
    ax.plot(list(fedavg_1.index), fedavg_1["val_accuracy"], '-', label='FedAvg-1')
    ax.plot(list(fedavg_2.index), fedavg_2["val_accuracy"], '-', label='FedAvg-2')
    ax.plot(list(fedavg_5.index), fedavg_5["val_accuracy"], '-', label='FedAvg-5')
    ax.plot(list(fedavg_10.index), fedavg_10["val_accuracy"], '-', label='FedAvg-10')
    plt.ylim(0, 1)
    legend = ax.legend(loc='center right')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.savefig("images/emnist-val-accuracy.png")


def emnist_experiment_validation_time_result_plot(clients_multiplier=1):
    global_sgd = pd.read_csv("src/experiments/history/global_history_ep10.csv")
    # TODO: This file was relabeled from `None'. Future code should output 1 for single epochs instead of `None'.
    fedavg_1 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp1.csv")
    fedavg_2 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp2.csv")
    fedavg_5 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp5.csv")
    fedavg_10 = pd.read_csv("src/experiments/history/history_fedavg_olr1.0_ilr0.01_b32_r100_rp10.csv")
    fig, ax = plt.subplots()
    N = 10
    colors = np.linspace(0, 1, N)
    cmap = ListedColormap(sns.color_palette("colorblind", N))
    ax.plot(list(global_sgd.index), global_sgd["time"].cumsum(), '-', label='Global')
    ax.plot(list(global_sgd.index)[0], global_sgd["time"].cumsum()[0], 'Dk', label='_nolegend_')
    ax.plot(list(fedavg_1.index), (fedavg_1["round_time"]*clients_multiplier).cumsum(), '-', label='FedAvg-1', markevery=[4])
    ax.plot(list(fedavg_1.index)[3], (fedavg_1["round_time"]*clients_multiplier).cumsum()[3], 'Dk', label='_nolegend_')
    ax.plot(list(fedavg_2.index), (fedavg_2["round_time"]*clients_multiplier).cumsum(), '-', label='FedAvg-2', markevery=[4])
    ax.plot(list(fedavg_2.index)[3], (fedavg_2["round_time"]*clients_multiplier).cumsum()[3], 'Dk', label='_nolegend_')
    ax.plot(list(fedavg_5.index), (fedavg_5["round_time"]*clients_multiplier).cumsum(), '-', label='FedAvg-5', markevery=[3])
    ax.plot(list(fedavg_5.index)[2], (fedavg_5["round_time"]*clients_multiplier).cumsum()[2], 'Dk', label='_nolegend_')
    ax.plot(list(fedavg_10.index), (fedavg_10["round_time"]*clients_multiplier).cumsum(), '-', label='FedAvg-10', markevery=[3])
    ax.plot(list(fedavg_10.index)[2], (fedavg_10["round_time"]*clients_multiplier).cumsum()[2], 'Dk', label='_nolegend_')
    plt.ylim(bottom=1)
    legend = ax.legend(loc='center right')
    plt.xlabel('Epochs (Global) / Rounds (FedAvg)')
    plt.ylabel('Time (s)')
    plt.savefig(f"images/emnist-train-time-{clients_multiplier}.png")
    plt.show()


def emnist_personal_finetune_acc_plot():
    global_sgd_acc = pd.read_csv("src/experiments/results/global_weights_ep09_acc0.99_acc.csv")
    fedavg_1_acc = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rpNone_acc.csv")
    fedavg_2_acc = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp2_acc.csv")
    fedavg_5_acc = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp5_acc.csv")
    fedavg_10_acc = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp10_acc.csv")

    fig, ax = plt.subplots()
    N = 10
    colors = np.linspace(0, 1, N)
    cmap = ListedColormap(sns.color_palette("colorblind", N))

    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], global_sgd_acc.mean(), "-", label="Global SGD")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_1_acc.mean(), "-", label="FedAvg-1")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_2_acc.mean(), "-", label="FedAvg-2")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_5_acc.mean(), "-", label="FedAvg-5")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_10_acc.mean(), "-", label="FedAvg-10")

    legend = ax.legend(loc='center right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"images/emnist-personal-acc.png")
    plt.show()


def emnist_personal_finetune_loss_plot():
    global_sgd_loss = pd.read_csv("src/experiments/results/global_weights_ep09_acc0.99_loss.csv")
    fedavg_1_loss = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rpNone_loss.csv")
    fedavg_2_loss = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp2_loss.csv")
    fedavg_5_loss = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp5_loss.csv")
    fedavg_10_loss = pd.read_csv("src/experiments/results/fedavg-weights_olr1.000_ilr0.010_b32_r100_ep1_rp10_loss.csv")

    fig, ax = plt.subplots()
    N = 10
    colors = np.linspace(0, 1, N)
    cmap = ListedColormap(sns.color_palette("colorblind", N))

    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], global_sgd_loss.mean(), "-", label="Global SGD")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_1_loss.mean(), "-", label="FedAvg-1")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_2_loss.mean(), "-", label="FedAvg-2")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_5_loss.mean(), "-", label="FedAvg-5")
    ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fedavg_10_loss.mean(), "-", label="FedAvg-10")

    legend = ax.legend(loc='center right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"images/emnist-personal-loss.png")
    plt.show()
