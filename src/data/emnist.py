import tensorflow as tf
import tensorflow_federated as tff


def get_dataset_size(dataset):
    size = 0
    for client in dataset.client_ids:
        size += len(dataset.create_tf_dataset_for_client(client))
    return size


def encode_dataset(dataset):
    return dataset.map(lambda x: (tf.expand_dims(x['pixels'], -1), tf.one_hot(x['label'], 10)), num_parallel_calls=4)


def load_flattened_dataset():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    onehot_train = encode_dataset(emnist_train.create_tf_dataset_from_all_clients().shuffle(buffer_size=256))
    onehot_test = encode_dataset(emnist_test.create_tf_dataset_from_all_clients())
    train_size = get_dataset_size(emnist_train)
    test_size = get_dataset_size(emnist_test)
    return (onehot_train, train_size), (onehot_test, test_size)
