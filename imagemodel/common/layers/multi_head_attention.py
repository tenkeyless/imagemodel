# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/multi_head_attention.py
import collections
import math
import string

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints, initializers, regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import advanced_activations, core, einsum_dense
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, math_ops, special_math_ops
from tensorflow.python.util.tf_export import keras_export

_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    (bs, <non-attention dims>, <attention dims>, num_heads, channels).
    bs and <non-attention dims> are treated as <batch dims>.
    The attention operations can be generalized:
    (1) Query-key dot product:
    (<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)
    (2) Combination:
    (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
    <query attention dims>, num_heads, channels)

    Args:
      rank: the rank of query, key, value tensors.
      attn_axes: a list/tuple of axes, [-1, rank), that will do attention.

    Returns:
      Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
            [target_notation[i] for i in batch_dims] +
            [target_notation[i] for i in attn_axes] +
            [source_notation[i] for i in attn_axes])
    dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                          product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                      target_notation)
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """멀티 헤드 어텐션 내부 프로젝션에 대한 einsum equation을 빌드합니다."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


@keras_export("keras.layers.MultiHeadAttention")
class MultiHeadAttention(Layer):
    """MultiHeadAttention 레이어.

    이것은 "Attention Is All You Need"을 기반으로 한 멀티 헤드 어텐션의 구현입니다.
    `query`, `key`, `value`가 같으면 셀프 어텐션입니다.
    `query`의 각 시간 단계(timestep)는 `key`의 해당 시퀀스에 attends 하고, 고정 너비 벡터를 반환합니다.

    이 레이어는 먼저 `query`, `key` 및 `value`을 프로젝션합니다.
    이들은 (효과적으로) `num_attention_heads` 길이의 텐서 리스트이며,
    해당 shape은 [batch_size, <query dimensions>, key_dim],
    [batch_size, <key/value dimensions>, key_dim],
    [batch_size, <key/value dimensions>, value_dim]입니다.

    그런 다음, 쿼리와 키 텐서가 내적(dot-producted)되고 스케일(scaled)됩니다.
    어텐션 확률을 얻기 위해, 소프트맥스됩니다.
    그런 다음 값 텐서는 이들 확률로 보간된 다음, 단일 텐서로 다시 연결(concatenated)됩니다.

    마지막으로, 마지막 차원이 value_dim인 결과 텐서는 선형 프로젝션을 취하고 반환할 수 있습니다.

    예제:

    어텐션 마스크를 사용하여 두 개의 시퀀스 입력에 대해 1D 교차 어텐션(1D cross-attention)을 수행합니다.
    헤드에 대한 추가적인 어텐션 가중치를 반환합니다.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    axes 2와 3에 대해 5D 입력 텐서에 걸쳐, 2D 셀프 어텐션을 수행합니다.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Arguments:
      num_heads: 어텐션 헤드의 수.
      key_dim: 쿼리 및 키에 대한 각 어텐션 헤드의 크기.
      value_dim: 값에 대한 각 어텐션 헤드의 크기. None이면, key_dim을 사용합니다.
      dropout: Dropout 확률.
      use_bias: Boolean, dense 레이어가 bias 벡터/행렬을 사용하는지 여부.
      output_shape: 배치 및 시퀀스 차원 외에, 예상되는 출력 텐서 shape 입니다.
        지정하지 않으면, 키 특성 차원으로 다시 프로젝션 됩니다.
      attention_axes: 어텐션이 적용되는 축(axes).
        `None`은 모든 축에 걸친 어텐션을 의미하지만, 배치, 헤드, 특성입니다.
      kernel_initializer: dense 레이어 커널을 위한 이니셜라이저.
      bias_initializer: dense 레이어 biases를 위한 이니셜라이저.
      kernel_regularizer: dense 레이어 커널을 위한 Regularizer.
      bias_regularizer: dense 레이어 biases를 위한 Regularizer.
      activity_regularizer: dense 레이어 activity를 위한 Regularizer.
      kernel_constraint: dense 레이어 커널을 위한 Constraint.
      bias_constraint: dense 레이어 커널을 위한 Constraint.

    Call arguments:
      query: `[B, T, dim]` shape의 쿼리(Query) `Tensor`.
      value: `[B, S, dim]` shape의 값(Value) `Tensor`
      key: `[B, S, dim]` shape의 Optional 키(key) Tensor.
        주어지지 않으면, 가장 일반적인 경우인, `key`와 `value` 모두에 `value`를 사용합니다.
      attention_mask: 특정 위치에 대한 어텐션을 방해하는, `[B, T, S]` shape의 boolean 마스크
      return_attention_scores: 출력이 어텐션 출력이어야 하는지 여부를 나타내는 boolean.
        True면, (attention_output, attention_scores) 입니다.
        False면, attention_output 입니다.
      training: 레이어가 트레이닝 모드(dropout 추가) 또는 추론 모드(dropout 없음)로 동작해야 하는지 여부를 나타내는 Python boolean.
        기본값은 상위 레이어/모델의 트레이닝 모드를 사용하거나, 상위 레이어가 없는 경우 False(추론)입니다.

    Returns:
      attention_output: shape `[B, T, E]`의 계산 결과입니다.
        여기서 `T`는 대상 시퀀스 shapes이고,
        `E`는 `output_shape`이 `None`인 경우, 쿼리 입력 마지막 차원입니다.
        그렇지 않으면, 멀티 헤드 출력이 `output_shape`에 지정된 shape으로 프로젝션됩니다.
      attention_scores: [Optional] 어텐션 axes에 걸친 멀티 헤드 어텐션 계수.
    """

    def __init__(
            self,
            num_heads,
            key_dim,
            value_dim=None,
            dropout=0.0,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim  # value 차원이 주어지지 않으면, key 차원을 사용
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        # attention_axes 어텐션 축이 따로 지정되어 있는 경우를 위한 지정
        if attention_axes is not None and not isinstance(attention_axes, collections.abc.Sized):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False

    def get_config(self):
        config = {
            "num_heads":
                self._num_heads,
            "key_dim":
                self._key_dim,
            "value_dim":
                self._value_dim,
            "dropout":
                self._dropout,
            "use_bias":
                self._use_bias,
            "output_shape":
                self._output_shape,
            "attention_axes":
                self._attention_axes,
            "kernel_initializer":
                initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                constraints.serialize(self._bias_constraint)
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _build_from_signature(self, query, value, key=None):
        """레이어와 변수를 빌드.

        메서드가 한 번 호출되면, self._built_from_signature가 `True`로 설정됩니다.

        Args:
          query: query 텐서 또는 TensorShape.
          value: value 텐서 또는 TensorShape.
          key: key 텐서 또는 TensorShape.
        """
        self._built_from_signature = True
        # query shape 지정
        if hasattr(query, "shape"):
            query_shape = tensor_shape.TensorShape(query.shape)
        else:
            query_shape = query
        # value shape 지정
        if hasattr(value, "shape"):
            value_shape = tensor_shape.TensorShape(value.shape)
        else:
            value_shape = value
        # key shape 지정. 명시되어 있지 않으면 value shape 사용
        if key is None:
            key_shape = value_shape
        elif hasattr(key, "shape"):
            key_shape = tensor_shape.TensorShape(key.shape)
        else:
            key_shape = key

        common_kwargs = dict(
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activity_regularizer=self._activity_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint)
        # 모든 설정 작업은, 나중에 eager 연산을 오염시킬 symbolic Tensors 생성을 방지하기 위해, `init_scope`에서 한 번만 수행되도록 발생해야 합니다.
        with tf_utils.maybe_init_scope(self):
            free_dims = query_shape.rank - 1
            einsum_equation, bias_axes, output_rank = _build_proj_equation(free_dims, bound_dims=1, output_dims=2)
            self._query_dense = einsum_dense.EinsumDense(
                    einsum_equation,
                    output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                    bias_axes=bias_axes if self._use_bias else None,
                    name="query",
                    **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                    key_shape.rank - 1,
                    bound_dims=1,
                    output_dims=2)
            self._key_dense = einsum_dense.EinsumDense(
                    einsum_equation,
                    output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                    bias_axes=bias_axes if self._use_bias else None,
                    name="key",
                    **common_kwargs)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                    value_shape.rank - 1, bound_dims=1, output_dims=2)
            self._value_dense = einsum_dense.EinsumDense(
                    einsum_equation,
                    output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._value_dim]),
                    bias_axes=bias_axes if self._use_bias else None,
                    name="value",
                    **common_kwargs)

            # 멀티 헤드 내적(dot product) 어텐션에 대한 어텐션 계산을 빌드합니다.
            # 이러한 계산은, 멀티 헤드 einsum 계산을 지원하면, keras 어텐션 레이어로 래핑될 수 있습니다.
            self._build_attention(output_rank)
            if self._output_shape:
                if not isinstance(self._output_shape, collections.abc.Sized):
                    output_shape = [self._output_shape]
                else:
                    output_shape = self._output_shape
            else:
                output_shape = [query_shape[-1]]
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                    free_dims,
                    bound_dims=2,
                    output_dims=len(output_shape))
            self._output_dense = einsum_dense.EinsumDense(
                    einsum_equation,
                    output_shape=_get_output_shape(output_rank - 1, output_shape),
                    bias_axes=bias_axes if self._use_bias else None,
                    name="attention_output",
                    **common_kwargs)

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        costomize attention computation to replace the default dot-product
        attention.

        Args:
          rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = (
            _build_attention_equation(rank, attn_axes=self._attention_axes))
        norm_axes = tuple(
                range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = advanced_activations.Softmax(axis=norm_axes)
        self._dropout_layer = core.Dropout(rate=self._dropout)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
            mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = array_ops.expand_dims(attention_mask, axis=mask_expansion_axes)
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
            self,
            query,
            key,
            value,
            attention_mask=None,
            training=None):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for customized
        attention implementation.

        Args:
          query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
          key: Projected key `Tensor` of shape `[B, T, N, key_dim]`.
          value: Projected value `Tensor` of shape `[B, T, N, value_dim]`.
          attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
            attention to certain positions.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = special_math_ops.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = special_math_ops.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False, training=None):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
                query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
