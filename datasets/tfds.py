import tensorflow_datasets as tfds


def get_tfds_datasets():
    return tfds.list_builders()


def get_oxford_iiit_pet_3_x():
    return tfds.load("oxford_iiit_pet:3.*.*", with_info=True)


def get_mnist():
    return tfds.load("mnist")


def get_voc2012():
    return tfds.load("voc2012", split=tfds.Split.TRAIN, batch_size=16)


def get_voc2007():
    return tfds.load("voc2007", split=tfds.Split.TRAIN, batch_size=16)
