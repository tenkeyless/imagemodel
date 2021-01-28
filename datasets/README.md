# Datasets

## TFDS

### `tfds.load`

Arguments

- `name`: `str`
  - `DatasetBuilder`에 등록된 이름(클래스 이름의 snake case 버전)
  - `BuilderConfigs`가 있는 데이터 세트에 대해, `'dataset_name'` 또는 `'dataset_name/config_name'`이 될 수 있습니다.
  - 편의상, 이 문자열에는 빌더에 대해 쉼표로 구분된 키워드 인수가 포함될 수 있습니다.
    - 예를 들어 `'foo_bar/a=True,b=3'`은 키워드 인수 `a=True` 및 `b=3`을 전달하는 `FooBar` 데이터 세트를 사용합니다. (구성이 있는 빌더의 경우, `'foo_bar/zoo/a=True,b=3'`를 사용하여, `'zoo'` 구성을 사용하고, 빌더 키워드 인수 `a=True` 및 `b=3`을 전달합니다.)
- `split`: `Optional[Tree[splits_lib.Split]]` = `None`
  - 로드할 데이터 분할 (예 : `'train'`, `'test'`, `['train', 'test']`, `'train[80%:]'`,...)
    - [분할 API 가이드](https://www.tensorflow.org/datasets/splits)를 참조하세요.
  - `None`이면, 모든 분할을 `Dict[Split, tf.data.Dataset]`로 반환합니다.
- `data_dir`: `Optional[str]` = `None`
  - 데이터를 읽고/쓸 디렉토리
  - 기본값은 환경 변수 `TFDS_DATA_DIR`(설정된 경우)의 값입니다.
    - 그렇지 않으면, `'~/tensorflow_datasets'`로 폴백(falls back)됩니다.
- `batch_size`: `Optional[int]` = `None`  
  - 설정된 경우, 예제에 배치 차원을 추가합니다.
  - 가변 길이 특성은 0으로 채워집니다. (0-padded)
  - `batch_size=-1`이면, 전체 데이터 세트를 `tf.Tensors`로 반환합니다.
- `shuffle_files`: `bool` = `False`
  - 입력 파일을 셔플하는 여부
  - 기본값은 `False` 입니다.
- `download`: `bool` = `True`
  - `tf.DatasetBuilder.as_dataset`를 호출하기 전에 [`tfds.core.DatasetBuilder.download_and_prepare`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder#download_and_prepare)를 호출할지 여부
    - `False`이면, 데이터가 `data_dir`에 있어야 합니다.
    - `True`이고, 데이터가 이미 `data_dir`에 있는 경우, `download_and_prepare`는 작동하지 않습니다.
- `as_supervised`: `bool` = `False`
  - `True`인 경우, 반환된 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)은 `builder.info.supervised_keys`에 따라 2-튜플 구조(입력, 레이블)를 갖습니다.
  - 기본값인 `False`이면, 반환된 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)에는 모든 특성이 포함된 딕셔너리가 있습니다.
- `decoders`: `Optional[TreeDict[decode.Decoder]]` = `None`
  - 디코딩을 커스터마이즈 할 수 있는 `Decoder` 객체의 중첩된 딕셔너리
  - 구조는 특성 구조와 일치해야 하지만, 커스터마이즈된 특성 키만 있으면 됩니다.
  - 자세한 내용은 [가이드](https://github.com/tensorflow/datasets/tree/master/docs/decode.md)를 참조하세요.
- `read_config`: `Optional[tfds.ReadConfig]` = `None`
  - [`tfds.ReadConfig`](https://www.tensorflow.org/datasets/api_docs/python/tfds/ReadConfig)
  - 입력 파이프라인을 구성하는 추가 옵션(예 : seed, num parallel reads 등)
- `with_info`: `bool` = `False`
  - `True`인 경우, [`tfds.load`](https://www.tensorflow.org/datasets/api_docs/python/tfds/load)는 튜플 ([`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), [`tfds.core.DatasetInfo`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo))을 반환하며, 후자는 빌더와 관련된 정보를 포함합니다.
- `builder_kwargs`: `Optional[Dict[str, Any]]` = `None`
  - [`tfds.core.DatasetBuilder`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder) 생성자에 전달할 키워드 인수입니다.
  - `data_dir`은 기본적으로 전달됩니다.
- `download_and_prepare_kwargs`: `Optional[Dict[str, Any]]` = `None`
  - `download=True`인 경우, [`tfds.core.DatasetBuilder.download_and_prepare`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder#download_and_prepare)에 전달되는 키워드 인수입니다.
  - 캐시된 데이터를 다운로드하고 추출할 위치를 제어할 수 있습니다.
  - 설정하지 않으면, `cache_dir` 및 `manual_dir`이 `data_dir`에서 자동으로 추론됩니다.
- `as_dataset_kwargs`: `Optional[Dict[str, Any]]` = `None`
  - [`tfds.core.DatasetBuilder.as_dataset`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder#as_dataset)에 전달된 키워드 인수입니다.
- `try_gcs`: `bool` = `False`
  - `True`인 경우, `tfds.load`는 로컬에서 빌드하기 전에, 데이터세트가 공개 GCS 버킷에 있는지 확인합니다.

Returns

- `ds`: [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
  - 요청된 데이터 세트
  - 또는 `split`이 `None`인 경우, `dict<key: tfds.Split, value: tf.data.Dataset>`.
  - `batch_size=-1`이면, [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)로 전체 데이터 세트가됩니다.
- `ds_info`: [`tfds.core.DatasetInfo`](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetInfo)
  - `with_info`가 `True`이면, [`tfds.load`](https://www.tensorflow.org/datasets/api_docs/python/tfds/load)는 데이터 세트 정보(버전, 특성, 분할, num_examples, ...)가 포함된 튜플 `(ds, ds_info)`을 반환합니다.
  - `ds_info` 객체는, 요청된 `split`에 관계없이, 전체 데이터 세트를 문서화합니다. 분할 관련 정보는 `ds_info.splits`에서 사용 가능합니다.
