import tensorflow as tf


def check_first_gpu():
    # GPU Setting
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Tensorflow use first gpu.
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
            print("GPU Setting Error")
