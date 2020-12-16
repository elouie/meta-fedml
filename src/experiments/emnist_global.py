import pandas as pd
import os

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

from src.data.emnist import load_flattened_dataset
from src.model.emnist import create_emnist_model
from src.utils.time_callback import TimeCallback


def run_global_emnist_experiment(lr=0.001, batch=32, epochs=10):
    train_dataset, train_size, val_dataset, val_size = load_flattened_dataset(batch)
    step_count = train_size
    val_step_count = val_size
    print(f"Testing with {step_count} training steps.")
    print(f"Testing with {val_step_count} validation steps.")

    # Callbacks during execution
    checkpoint_path = "src/experiments/checkpoints/global_weights_ep{epoch:02d}_acc{val_accuracy:.2f}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    time_callback = TimeCallback()
    callbacks_list = [checkpoint, time_callback]

    model = create_emnist_model()
    opt = SGD(learning_rate=lr, momentum=0.99)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_dataset.repeat(epochs),
                        steps_per_epoch=step_count,
                        validation_data=val_dataset.repeat(epochs).prefetch(1),
                        validation_steps=val_step_count,
                        callbacks=callbacks_list,
                        epochs=10,
                        verbose=2)
    print(history.history)
    pd.DataFrame.from_dict(history.history).to_csv(f'src/experiments/history/global_history_ep{epochs:02d}.csv', index=False)
