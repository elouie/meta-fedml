import pandas as pd
import os

from math import ceil
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

from src.data.emnist import load_flattened_dataset
from src.model.emnist import create_emnist_model


def run_global_emnist_experiment(lr=0.001, batch=32, epochs=10):
    train_dataset, val_dataset = load_flattened_dataset()
    step_count = ceil(train_dataset[1] / batch)
    val_step_count = ceil(val_dataset[1] / batch)

    # Callbacks during execution
    checkpoint_path = "checkpoints/global-weights-ep{epoch:02d}-acc{val_accuracy:.2f}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    model = create_emnist_model()
    opt = SGD(learning_rate=lr, momentum=0.99)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_dataset[0].batch(batch, drop_remainder=True).repeat(epochs).prefetch(1),
                        validation_data=val_dataset[0].batch(batch, drop_remainder=True).repeat(epochs).prefetch(1),
                        validation_steps=val_step_count,
                        steps_per_epoch=step_count,
                        callbacks=callbacks_list,
                        epochs=10)
    pd.DataFrame.from_dict(history.history).to_csv('history/global-history.csv', index=False)
