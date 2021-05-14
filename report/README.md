# Report

## 1. Merge Images

`merge_pdf.sh`

- Options
  - `-f`: "[Folder Path]@[Label on Image]"
    - Multiple `-f` options can be used.
    - The order of image placement is in the order of `-f`.
    - Lists the file names that exist in the folder of the first `-f` option applied to the folder of the later `-f` options.
    - If the matching file does not exist in later folders, an error may occur.
    - Example) "my_image@MY IMAGE"
  - `-t`: "[Target file path]"
    - Target file path and file name
    - Example) "/folder/path/my_result.pdf"

- Requirements
  - [imagemagick](https://imagemagick.org/)

- Usage
  - `merge_pdf.sh` -f "[Folder Path]@[Label on Image]" -t "[Target file path]"

- Example
  
  ```shell
  ./merge_pdf.sh \
  -f "[input] current image@[input] current image" \
  -f "[input] p1 label@[input] prev label" \
  -f "[output] _predict__model_ref_local_tracking_model_022__run_leetaekyu_20210127_091102@[predict]" \
  -f "[target] current label@[target] current label" \
  -t "predict__model_ref_local_tracking_model_022__run_leetaekyu_20210127_091102.pdf"
  ```

## 2. Compare Image

`compare.sh`

- Options
  - `-f`: "[Folder Path]"
    - Compare from folder
    - Example) "my_image_target"
  - `-t`: "[Folder Path]"
    - Compare with folder
    - Example) "my_image_predict"
  - `-s`: "[Target Folder]"
    - Path for the target result files

- Requirements
  - [imagemagick](https://imagemagick.org/)

- Usage
  - `compare.sh` -f "[Folder Path]" -t "[Folder Path]" -s "[Folder Path]"

- Example
  
  ```shell
  ./compare.sh \
  -f "/Users/tklee/workspace/cell segmentation/results/[target] current label" \
  -t "/Users/tklee/workspace/cell segmentation/results/predict_testset__model_ref_local_tracking_model_033__run_leetaekyu_20210317_135008/predict_result/images" \
  -s "/Users/tklee/workspace/cell segmentation/results/predict_testset__model_ref_local_tracking_model_033__run_leetaekyu_20210317_135008/compare_with_target"
  ```

## 3. Get Results

### 3.1. Training Result

`get_training_result.sh`

Get 1) a training info, model structure, sample images and 2) a log for TensorBoard.

- Requirements
  - gsutil

- Usage
  - `get_training_result.sh` "[TRAINING_ID]" "[TARGET_FOLDER]"

- Example

  ```shell
  ./get_training_result.sh training__model_ref_local_tracking_model_011__run_leetaekyu_20210109_012720 ~/Downloads/result/test
  ```

### 3.2. Test Result

`get_test_result.sh`

Get 1) a test info.

- Requirements
  - gsutil

- Usage
  - `get_test_result.sh` "[TEST_ID]" "[TARGET_FOLDER]"

- Example

  ```shell
  ./get_test_result.sh test__model_ref_local_tracking_model_011__run_leetaekyu_20210109_012720 ~/Downloads/result/test
  ```

### 3.3. Predict Result

`get_predict_result.sh`

Get 1) a predicted results.

- Requirements
  - gsutil

- Usage
  - `get_predict_result.sh` "[PREDICTION_ID]" "[TARGET_FOLDER]"

- Example

  ```shell
  ./get_predict_result.sh predict_testset__model_ref_local_tracking_model_011__run_leetaekyu_20210109_012720 ~/Downloads/result/test
  ```
