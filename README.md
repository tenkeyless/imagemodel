# imagemodel

Deep learning on TensorFlow, Keras for CPU, GPU and TPU.

"Image segmentation (binary)", "Image segmentation (category)" and "classification" are provided.

For a description of each image deep learning, please refer to the "README.md" file in the corresponding folder.

* Image segmentation (binary)
  * `imagemodel/binary_segmentations`
* Image segmentation (category)
  * `imagemodel/category_segmentations`
  * Models
    * U-Net
    * U-Net Level
    * U-Net based on MobileNet V2
* Image classification
  * `imagemodel/classification`

## Prerequisite

This project runs based on Docker.

### Docker

1. Build `Dockerfile`.

    ```shell
    cd code/segmentations
    docker build . -t segmentations/tkl:1.0
    ```

    * [Result] Check docker image

        ```shell
        $ docker images
        segmentations/tkl       1.0                               f1a65192f6b7        5 days ago          8.3GB
        nvidia/cuda             11.0-cudnn8-runtime-ubuntu18.04   848be2582b0a        12 days ago         3.6GB
        nvidia/cuda             10.2-base                         038eb67e1704        2 weeks ago         107MB
        nvidia/cuda             latest                            752312fac010        3 weeks ago         4.69GB
        nvidia/cuda             10.0-base                         0f12aac8787e        3 weeks ago         109MB
        ...
        ```

2. Run docker image as bash.

    At [This project] folder. (`$(pwd)`)

    ```shell
    mkdir [result folder]
    mkdir [tensorflow dataset folder]
    cd [this code folder]
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/segmentations \
        -v [result folder]:/category_segmentations_results \
        -v [tensorflow dataset folder]:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/segmentations" \
        segmentations/tkl:1.0
    ```

    Real example.

    ```shell
    mkdir ~/category_segmentations_results
    mkdir ~/tensorflow_datasets
    cd code/segmentations
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/segmentations \
        -v ~/category_segmentations_results:/category_segmentations_results \
        -v ~/tensorflow_datasets:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/segmentations" \
        segmentations/tkl:1.0
    ```

### Tips for docker

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
