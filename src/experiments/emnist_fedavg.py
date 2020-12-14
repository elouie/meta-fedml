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
from src.utils.federated_utils import build_categorical_model_fn


def run_fedavg_emnist_experiment(inner_lr=0.01, outer_lr=1.0, batch=32, rounds=50, epochs=1):
    train_dataset, val_dataset = load_federated_dataset(batch)

    print("Building model function.", flush=True)
    # This creates a consistent model to be created per client to place the global weights into.
    build_fn = build_categorical_model_fn(create_emnist_model, train_dataset[0])

    print("Building metrics and process.", flush=True)
    federated_eval = tff.learning.build_federated_evaluation(build_fn)
    history = defaultdict(list)

    # This creates a "process" that executes rounds using the supplied model generator and optimizers.
    fedavg_process = tff.learning.build_federated_averaging_process(
        build_fn,
        client_optimizer_fn=lambda: SGD(learning_rate=inner_lr),
        server_optimizer_fn=lambda: SGD(learning_rate=outer_lr, momentum=0.99))

    print("Starting training.", flush=True)
    state = fedavg_process.initialize()
    batched_dataset = train_dataset
    for i in range(rounds):
        print("Starting roung ", i)
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
        print('round {:2d}, metrics={}'.format(i, train_metrics))
        val_metrics = federated_eval(state.model, val_dataset)
        for k in val_metrics:
            history["val_" + k].append(val_metrics[k])
        temp_model = create_emnist_model()
        state.model.assign_weights_to(temp_model)
        temp_model.save_weights(f"checkpoints/global-weights_olr{outer_lr:.3f}_ilr{inner_lr:.3f}_b{batch}_r{rounds}_ep{epochs}_acc{val_accuracy:.2f}_i.hdf")

    pd.DataFrame.from_dict(history).to_csv(f'history/history_fedavg_olr{outer_lr}_ilr{inner_lr}_b{batch}_r{rounds}.csv', index=False)
