from typing import List

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from imagemodel.common.utils.compare import compare_func


def model_is_equal(model1: Model, model2: Model) -> bool:
    """
    Check TF Model equality.

    - Total number of parameters
    - Number of layers
    - Number of parameters in all layers
    - Shape of inputs
    - Shape of outputs

    Parameters
    ----------
    model1
    model2

    Returns
    -------
    bool
    """
    check_list: List[bool] = [compare_func(get_total_number_of_parameters, model1, model2),
                              compare_func(get_number_of_layers, model1, model2),
                              compare_func(get_parameters_from_all_layers, model1, model2),
                              compare_func(get_inputs_shape, model1, model2),
                              compare_func(get_outputs_shape, model1, model2)]
    return all(check_list)


def get_number_of_layers(model: Model) -> int:
    return len(model.layers)


def get_total_number_of_parameters(model: Model) -> int:
    return model.count_params()


def get_parameters_from_all_layers(model: Model) -> List[int]:
    layers: List[Layer] = model.layers
    return list(map(get_number_of_parameters, layers))


def get_inputs_shape(model: Model) -> List[List[int]]:
    return list(map(lambda el: el.shape.as_list(), model.inputs))


def get_outputs_shape(model: Model) -> List[List[int]]:
    return list(map(lambda el: el.shape.as_list(), model.outputs))


def get_number_of_parameters(layer: Layer) -> int:
    return layer.count_params()
