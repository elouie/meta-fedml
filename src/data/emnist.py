import random
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def get_dataset_size(dataset):
    size = 0
    for client in dataset.client_ids:
        size += len(dataset.create_tf_dataset_for_client(client))
    return size


def encode_element(x):
    return (tf.expand_dims(x['pixels'], -1), tf.one_hot(x['label'], 10))


def encode_dataset(dataset, batch):
    return dataset.shuffle(2*batch, reshuffle_each_iteration=True) \
        .batch(batch, drop_remainder=True) \
        .map(encode_element, num_parallel_calls=4) \
        .prefetch(1)


def load_flattened_dataset(batch):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    onehot_train = encode_dataset(emnist_train.create_tf_dataset_from_all_clients())
    onehot_test = encode_dataset(emnist_test.create_tf_dataset_from_all_clients())
    train_size = get_dataset_size(emnist_train)
    test_size = get_dataset_size(emnist_test)
    return (onehot_train, train_size), (onehot_test, test_size)


def make_federated_data(train_data, test_data, batch):
    print("Beginning dataset generation.", flush=True)
    # A pregenerated list of clients into 30 (roughly even) splits.
    # We can't use the full split as TFF is _really slow_ with large numbers of
    # clients in simulation.
    client_ids_split = np.load("test_split.npy", allow_pickle=True)
    train_datasets = []
    train_sizes = []
    test_datasets = []
    test_sizes = []
    for client_ids in client_ids_split:
        train_dataset = train_data.create_tf_dataset_for_client(client_ids[0])
        test_dataset = test_data.create_tf_dataset_for_client(client_ids[0])
        for i in range(1, len(client_ids)):
            train_dataset = train_dataset.concatenate(train_data.create_tf_dataset_for_client(client_ids[i]))
            test_dataset = test_dataset.concatenate(test_data.create_tf_dataset_for_client(client_ids[i]))
        train_datasets.append(encode_dataset(train_dataset, batch))
        test_datasets.append(encode_dataset(test_dataset, batch))
        train_sizes.append(len(train_dataset))
        test_sizes.append(len(test_dataset))
    val_dataset = encode_dataset(test_data.create_tf_dataset_from_all_clients(), batch)
    val_size = get_dataset_size(test_data)
    return train_datasets, train_sizes, test_datasets, test_sizes, val_dataset, val_size


def load_federated_dataset(batch):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    per_client_train = [encode_dataset(emnist_train.create_tf_dataset_for_client(i), batch) for i in emnist_train.client_ids]
    per_client_test = [encode_dataset(emnist_test.create_tf_dataset_for_client(i), batch) for i in emnist_test.client_ids]
    return make_federated_data(emnist_train, emnist_test, batch)
