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
    - If you have space in folder or file name, you need to write like this.
      - 'folder\ or/file\ name\ with\ space.pdf'
    - Example) "/folder/path/my_result.pdf"

- Requirements
  - [imagemagick](https://imagemagick.org/)

- Usage
  - `merge_pdf.sh` -f "[Folder Path]@[Label on Image]" -t "[Target file path]"

- Example
  
  ```shell
  ./merge_pdf.sh \
  -f "[folder 1]@[text label below image 1]" \
  -f "[folder 2]@[text label below image 2]" \
  -f "[folder 3]@[text label below image 3]" \
  -t "[target folder]/[filename.pdf]"
  ```

## 2. Get Results

### 2.1. Training Result

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

### 2.2. Test Result

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

### 2.3. Predict Result

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
