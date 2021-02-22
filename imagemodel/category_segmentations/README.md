# Category Segmentation

Segmentation indicating which category each pixel in the image belongs to.

## Model

* Inputs
  * [`width`, `height`, 3] or [`width`, `height`, 1]
    * RGB Image or Grayscale Image
* Outputs
  * [`width`, `height`, `number of category`]
    * A category for each pixel in image.

## Prerequisite

### Docker

1. Build Dockerfile.

    ```shell
    cd code/imagemodel
    docker build . -t imagemodel/tkl:1.0
    ```

    * [Result] Check docker image

        ```shell
        $ docker images
        imagemodel/tkl          1.0                               f1a65192f6b7        5 days ago          8.3GB
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
        -v $(pwd):/imagemodel \
        -v [result folder]:/category_segmentations_results \
        -v [tensorflow dataset folder]:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/imagemodel" \
        imagemodel/tkl:1.0
    ```

    Real example.

    ```shell
    mkdir ~/category_segmentations_results
    mkdir ~/tensorflow_datasets
    cd code/imagemodel
    docker run \
        --gpus all \
        -it \
        --rm \
        -u $(id -u):$(id -g) \
        -v /etc/localtime:/etc/localtime:ro \
        -v $(pwd):/imagemodel \
        -v ~/category_segmentations_results:/category_segmentations_results \
        -v ~/tensorflow_datasets:/tensorflow_datasets \
        -p 6006:6006 \
        --workdir="/imagemodel" \
        imagemodel/tkl:1.0
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

* Using ssh -L

    ```shell
    ssh -L 127.0.0.1:16006:0.0.0.0:6006 username@server
    ```

  * As mac client, I use "SSH Tunnel Manager" app to connect server.

### `Tmux` for session

* I use `tmux` or `screen` for session.

## Run

### U-Net

* Run training on docker container for `oxford_iiit_pet_v3` training dataset.

    ```shell
    python imagemodel/category_segmentations/run/train.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet \
      --model_option 'input_shape@(128, 128, 3)' \
      --model_option 'output_channels@3' \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --training_epochs 20 \
      --val_freq 1 \
      --run_id tkl_20210220_174938 \
      --optimizer adam2 \
      --losses 'sparse_categorical_crossentropy',1.0 \
      --metrics accuracy
    ```

* Run test on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/test.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --model_weight_path /category_segmentations_results/save/weights/training__model_unet__run_tkl_20210220_174938.epoch_08 \
      --run_id tkl_20210220_175156 \
      --optimizer adam2 \
      --losses 'sparse_categorical_crossentropy',1.0 \
      --metrics accuracy
    ```

* Run prediction on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/predict_on_test_dataset.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --model_weight_path /category_segmentations_results/save/weights/training__model_unet__run_tkl_20210220_174938.epoch_08 \
      --run_id tkl_20210220_175256
    ```

### U-Net Level

* Run training on docker container for `oxford_iiit_pet_v3` training dataset.

    ```shell
    python imagemodel/category_segmentations/run/train.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet_level \
      --model_option 'input_shape@(128, 128, 3)' \
      --model_option 'level@4' \
      --model_option 'output_channels@3' \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --training_epochs 20 \
      --val_freq 1 \
      --run_id tkl_20210221_210326 \
      --optimizer adam2 \
      --losses 'sparse_categorical_crossentropy',1.0 \
      --metrics accuracy
    ```

* Run test on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/test.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet_level \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --model_weight_path /category_segmentations_results/save/weights/training__model_unet_level__run_tkl_20210221_210326.epoch_07 \
      --run_id tkl_20210220_175156 \
      --optimizer adam2 \
      --losses 'sparse_categorical_crossentropy',1.0 \
      --metrics accuracy
    ```

* Run prediction on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/predict_on_test_dataset.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet_level \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --model_weight_path /category_segmentations_results/save/weights/training__model_unet_level__run_tkl_20210221_210326.epoch_07 \
      --run_id tkl_20210220_175256
    ```

### U-Net based on MobileNet V2

* Run training on docker container for `oxford_iiit_pet_v3` training dataset.

    ```shell
    python imagemodel/category_segmentations/run/train.py \
      --dataset oxford_iiit_pet_v3 \
      --model_name unet_based_mobilenetv2 \
      --model_option 'output_channels@3' \
      --result_base_folder /category_segmentations_results \
      --batch_size 8 \
      --training_epochs 20 \
      --val_freq 1 \
      --run_id tkl_20210221_205637 \
      --optimizer adam2 \
      --losses 'sparse_categorical_crossentropy_from_logits',1.0 \
      --metrics accuracy
    ```

* Run test on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/test.py \
        --dataset oxford_iiit_pet_v3 \
        --model_name unet_based_mobilenetv2 \
        --result_base_folder /category_segmentations_results \
        --batch_size 8 \
        --model_weight_path /category_segmentations_results/save/weights/training__model_unet_based_mobilenetv2__run_tkl_20210221_205637.epoch_06 \
        --run_id tkl_20210221_205609 \
        --optimizer adam2 \
        --losses 'sparse_categorical_crossentropy_from_logits',1.0 \
        --metrics accuracy
    ```

* Run prediction on docker container for `oxford_iiit_pet_v3` test dataset.

    ```shell
    python imagemodel/category_segmentations/run/predict_on_test_dataset.py \
        --dataset oxford_iiit_pet_v3 \
        --model_name unet_based_mobilenetv2 \
        --result_base_folder /category_segmentations_results \
        --batch_size 8 \
        --model_weight_path /category_segmentations_results/save/weights/training__model_unet_based_mobilenetv2__run_tkl_20210221_205637.epoch_06 \
        --run_id tkl_20210221_205753
    ```
