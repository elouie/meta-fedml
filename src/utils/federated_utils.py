import tensorflow_federated as tff
from tensorflow.keras import metrics, losses


def build_categorical_model_fn(model_builder, dataset):
    """
        Builds a model construction function for iterative processes.

        :param model: A model to be trained.
        :type model: tensorflow.keras.Model
        :param dataset: A dataset that specifies the element inputs.
        :type dataset: tensorflow.data.Dataset
        :return: A function that constructs a new federated model.
        :rtype: tff.learning.Model
    """
    def model_fn():
        keras_model = model_builder()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=dataset.element_spec,
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalCrossentropy(), metrics.CategoricalAccuracy()])
    return model_fn
