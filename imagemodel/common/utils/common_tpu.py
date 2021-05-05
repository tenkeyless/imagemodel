import subprocess

import tensorflow as tf


def create_tpu(
        tpu_name: str,
        ctpu_zone: str,
        # range: str = "10.240.0.0/29",
        accelerator_type: str = "v3-8",
        version="2.3.1"):
    subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "create",
                tpu_name,
                "--zone",
                ctpu_zone,
                # "--range",
                # range,
                "--accelerator-type",
                accelerator_type,
                "--version",
                version,
                "--preemptible",
                "--quiet"])


def tpu_initialize(tpu_address: str, tpu_zone: str):
    """
    Initializes TPU for TF 2.x training.

    Parameters
    ----------
    tpu_address : str
        bns address of master TPU worker.
    tpu_zone : str

    Returns
    -------
    TPUClusterResolver
        A TPUClusterResolver.
    """
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address, zone=tpu_zone)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    # return tf.distribute.experimental.TPUStrategy(resolver)
    return tf.distribute.TPUStrategy(resolver)


def delete_tpu(tpu_name: str, ctpu_zone: str):
    subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "delete",
                tpu_name,
                "--zone",
                ctpu_zone,
                "--quiet"])
