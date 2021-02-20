import subprocess


def create_tpu(
    tpu_name: str,
    ctpu_zone: str,
    # range: str = "10.240.0.0/29",
    accelerator_type: str = "v3-8",
    version="2.3.1",
):
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
            "--quiet",
        ]
    )


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
            "--quiet",
        ]
    )
