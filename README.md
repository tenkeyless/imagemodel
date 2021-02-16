# Segmentations

Image segmentations with Keras, TensorFlow for CPU, GPU and TPU.

## Requirements

* Python &ge; 3.7
* TensorFlow &ge; 2.4.0
* Keras &ge; 2.4.0

## With docker

### Generate docker image

* Change folder

    ```shell
    cd code/segmentations
    ```

* Build docker

    ```shell
    docker build . -t segmentations/tkl:1.0
    ```

* Docker image

    ```shell
    docker images

    <none>                  <none>                            53bf6b5d0f6a        44 hours ago        5.2GB
    nvidia/cuda             11.0-cudnn8-runtime-ubuntu18.04   848be2582b0a        12 days ago         3.6GB
    nvidia/cuda             10.2-base                         038eb67e1704        2 weeks ago         107MB
    nvidia/cuda             latest                            752312fac010        3 weeks ago         4.69GB
    nvidia/cuda             10.0-base                         0f12aac8787e        3 weeks ago         109MB
    ```

### Run docker image as container

* Need to make a [tensorflow dataset folder] before use tensorflow dataset.

    ```shell
    mkdir ~/tensorflow_datasets
    ```
  
  * Check your permission of [tensorflow dataset folder].

    ```shell
    ls -al

    ...
    drwxr-xr-x  2 tklee tklee  4096 10월  7 14:49 Templates
    drwxrwxr-x  2 root  root  4096  2월 14 19:09 tensorflow_datasets
    drwxr-xr-x  2 tklee tklee  4096 10월  7 14:49 Videos
    ...
    ```

    If owner of [tensorflow dataset folder] is root, then you need to change it.

    ```shell
    sudo chown -R [user]:[user] tensorflow_datasets/
    ```

    ```shell
    sudo chown -R tklee:tklee tensorflow_datasets/
    ```

* Run docker image as bash

    At [This project] folder. (`$(pwd)`)

    ```shell
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/segmentations \
        -v [result folder]:/segmentations_results \
        -v [tensorflow dataset folder]:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/segmentations" \
        segmentations/tkl:1.0
    ```

    ```shell
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/segmentations \
        -v ~/segmentations_results:/segmentations_results \
        -v ~/tensorflow_datasets:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/segmentations" \
        segmentations/tkl:1.0
    ```

### Training, test, prediction on docker container

* Run training on docker container

    ```shell
    python segmentations/run/train.py \
        --dataset oxford_iiit_pet_v3 \
        --model_name unet_based_mobilenetv2 \
        --result_base_folder /segmentations_results \
        --model_option 'output_channels@3' \
        --losses 'sparse_categorical_crossentropy_from_logits',1.0 \
        --metrics accuracy \
        --optimizer adam2
    ```

* Run test on docker container

    ```shell
    python segmentations/run/test.py \
        --dataset oxford_iiit_pet_v3 \
        --model_name unet_based_mobilenetv2 \
        --result_base_folder /segmentations_results \
        --batch_size 8 \
        --model_weight_path /segmentations_results/save/weights/training__model_unet_based_mobilenetv2__run_20210214_103920.epoch_06 \
        --run_id "20210216_110359" \
        --losses 'sparse_categorical_crossentropy_from_logits',1.0 \
        --metrics accuracy \
        --optimizer adam2
    ```

* Run prediction for test dataset on docker container

    ```shell
    python segmentations/run/predict_on_test_dataset.py \
        --dataset oxford_iiit_pet_v3 \
        --model_name unet_based_mobilenetv2 \
        --result_base_folder /segmentations_results \
        --batch_size 2 \
        --model_weight_path /segmentations_results/save/weights/training__model_unet_based_mobilenetv2__run_20210214_103920.epoch_06 \
        --run_id "20210216_140754"
    ```

## Tips for Docker

* Detach from docker container

    Ctrl+p, Ctrl+q

* Attach to docker container again

    Show running docker containers.

    ```shell
    $ docker ps
    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
    4c25ce8443e6        d59e4204feec        "/bin/bash"         4 hours ago         Up 4 hours                              zen_mendeleev
    ```

    Attach to container 4c25ce8443e6(Container id).

    ```shell
    docker attach 4c25ce8443e6
    docker attach $(docker ps -aq)
    ```

### (Optional) Tensorboard

* Run tensorboard on docker container

    ```shell
    docker exec [container id] tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    ```

    Real example

    ```shell
    docker exec 7f1840636c9d tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    docker exec $(docker ps -aq) tensorboard --logdir ./save/tf_logs/ --host 0.0.0.0 &
    ```

* Using ssh -L

    ```shell
    ssh -L 127.0.0.1:16006:0.0.0.0:6006 username@server
    ```

  * As mac client, I use "SSH Tunnel Manager" app to connect server.
