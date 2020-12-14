import random
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
    return dataset.shuffle(2*batch) \
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
    # Really, a fixed set of partitioned client ids needs to exist.
    print("Beginning dataset generation.", flush=True)
    client_ids = train_data.client_ids
    random.seed(0)
    random.shuffle(client_ids)
    train_dataset = None
    test_dataset = None
    train_datasets = []
    test_datasets = []
    for i, cid in enumerate(client_ids):
        if i % 50 == 0:
            if train_dataset is not None:
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
            train_dataset = encode_dataset(train_data.create_tf_dataset_for_client(cid), batch)
            test_dataset = encode_dataset(test_data.create_tf_dataset_for_client(cid), batch)
        else:
            train_dataset.concatenate(encode_dataset(train_data.create_tf_dataset_for_client(cid), batch))
            test_dataset.concatenate(encode_dataset(test_data.create_tf_dataset_for_client(cid), batch))
    return train_datasets, test_datasets


def load_federated_dataset(batch):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    per_client_train = [encode_dataset(emnist_train.create_tf_dataset_for_client(i), batch) for i in emnist_train.client_ids]
    per_client_test = [encode_dataset(emnist_test.create_tf_dataset_for_client(i), batch) for i in emnist_test.client_ids]
    return make_federated_data(emnist_train, emnist_test, batch)
