import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import SGD

from src.data.emnist import load_federated_dataset


def run_personalized_experiment(model_path, results_path, lr=0.0001, epochs=10, batch=32):
    """
        This experiment takes a pre-built model and fine-tunes the model
        to each train split, testing against the equivalent test set.
        The purpose is to elucidate whether a FedAvg trained model can
        outperform a global model in per-client performance. The idea is
        drawn from MAML, which uses the same outer-inner loop optimization
        for quick adaptation to new tasks in few epochs.

        Steps:
            1. Load model
            2. For each test split
            3.     Copy model
            4.     Fine-tune to test split
            5.     Aggregate validation statistics.

        :param model_path: The path of the model to be tested.
        :type model_path: str
        :param results_path: Where to place a CSV of the results.
        :type results_path: str
        :param lr: The learning rate the fine-tuning should use.
        :type lr: float
        :param epochs: The number of epochs during fine-tuning.
        :type epochs: int
    """

    # 1. Load model
    train_datasets, train_sizes, test_datasets, test_sizes, _, _ = load_federated_dataset(batch)

    # Create storage for results
    val_acc = np.zeros((len(train_sizes), epochs))
    val_loss = np.zeros((len(train_sizes), epochs))
    for i, train_dataset in enumerate(train_datasets):
        train_size = train_sizes[i]
        test_dataset = test_datasets[i]
        test_size = test_sizes[i]

        # Theoretically, we could use model.clone_model(), but experimentally it seems
        # to keep the weights from each fine-tune, so just reload, even though it's
        # slower.
        test_model = tf.keras.models.load_model(model_path)

        # The TF docs suggest the optimizer is saved with the model. Using this is
        # _a bad idea_. The optimizer saved is the global optimizer, which has a
        # learning rate of 1. Plus, further optimization over fine-tuning rate.
        opt = SGD(learning_rate=lr, momentum=0.99)
        test_model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        history = test_model.fit(train_dataset.repeat(epochs),
                            steps_per_epoch=train_size,
                            validation_data=test_dataset.repeat(epochs),
                            validation_steps=test_size,
                            epochs=epochs)
        val_acc[i, :] = history.history['val_accuracy']
        val_loss[i, :] = history.history['val_loss']
    pd.DataFrame(val_acc).to_csv(results_path + '_acc.csv', index=False)
    pd.DataFrame(val_loss).to_csv(results_path + '_loss.csv', index=False)
