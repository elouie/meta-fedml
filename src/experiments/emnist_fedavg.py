import pandas as pd
import os
import tensorflow_federated as tff
import time

from collections import defaultdict
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy

from src.data.emnist import load_federated_dataset
from src.model.emnist import create_emnist_model
from src.utils.federated_utils import build_categorical_model_fn, save_checkpoint


def run_fedavg_emnist_experiment(inner_lr=0.01, outer_lr=1.0, batch=32, rounds=100, epochs=1, repeats=None, early_stop_count=3):
    save_path = f"checkpoints/fedavg-weights_olr{outer_lr:.3f}_ilr{inner_lr:.3f}_b{batch}_r{rounds}_ep{epochs}_rp{repeats}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Loading datasets.")
    train_dataset, train_sizes, test_dataset, test_sizes, val_dataset, val_size = load_federated_dataset(batch)

    print("Building model function.", flush=True)
    # This creates a consistent model to be created per client to place the global weights into.
    build_fn = build_categorical_model_fn(create_emnist_model, train_dataset[0])

    print("Building metrics and process.", flush=True)
    # federated_eval = tff.learning.build_federated_evaluation(build_fn)
    history = defaultdict(list)

    # This creates a "process" that executes rounds using the supplied model generator and optimizers.
    fedavg_process = tff.learning.build_federated_averaging_process(
        build_fn,
        client_optimizer_fn=lambda: SGD(learning_rate=inner_lr),
        server_optimizer_fn=lambda: SGD(learning_rate=outer_lr, momentum=0.99))

    print("Starting training.", flush=True)
    state = fedavg_process.initialize()

    # If we want to emulate inner loop epochs, dataset provides the .repeat()
    # function to iterate multiple times on the dataset.
    batched_dataset = train_dataset
    if repeats is not None:
        print("Using repeats for emulating inner epochs: ", repeats)
        batched_dataset = []
        for dataset in train_dataset:
            batched_dataset.append(dataset.repeat(repeats))
    best_accuracy = 0
    bad_rounds = 0
    for i in range(rounds):
        print("Starting round ", i)
        # Training
        start = time.time()
        state, train_metrics = fedavg_process.next(state, batched_dataset)
        end = time.time()
        avg_round_time = (end - start)/len(train_dataset)
        print("Round time: ", avg_round_time)
        history["round_time"].append(avg_round_time)
        for k in train_metrics["train"]:
            history["train_" + k].append(train_metrics["train"][k])
        print("Metrics:", train_metrics["train"])

        # Now we generate a model from state to validate the dataset, as TFF
        # does not handle the Dataset type well (ie. the dataset does not have a
        # size and TF / TFF don't permit that, which the federated_eval requires)
        temp_model = create_emnist_model()
        state.model.assign_weights_to(temp_model)
        opt = SGD(learning_rate=inner_lr, momentum=0.99)
        temp_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        metrics = temp_model.evaluate(val_dataset, batch_size=batch, steps=val_size)
        for j, metric_label in enumerate(temp_model.metrics_names):
             history["val_" + metric_label].append(metrics[j])
        if metrics[1] > best_accuracy:
            bad_rounds = 0
            best_accuracy = metrics[1]
            print(f"Best models at epoch {i} with accuracy {best_accuracy}")
            temp_model.save(save_path + f"{i}")
        else:
            bad_rounds += 1
            if bad_rounds == early_stop_count:
              print(f"Ending early at round {i}")
              break
    pd.DataFrame.from_dict(history).to_csv(f'history/history_fedavg_olr{outer_lr}_ilr{inner_lr}_b{batch}_r{rounds}_rp{repeats}.csv', index=False)
