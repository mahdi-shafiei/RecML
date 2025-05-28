# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Layer utilities."""

from typing import Any
import keras

Tensor = Any


def clone_initializer(initializer: Any) -> Any:
  """Clones an initializer."""
  if isinstance(initializer, keras.initializers.Initializer):
    return initializer.clone()
  return initializer


def make_attention_mask(mask: Tensor, dtype: str = "float32") -> Tensor:
  """Creates a 3D self-attention mask from a padding mask."""
  # Element wise pairwise function on [B, L, 1], [B, 1, L].
  attention_mask = keras.ops.multiply(
      keras.ops.expand_dims(mask, axis=-1),
      keras.ops.expand_dims(mask, axis=-2),
  )
  return keras.ops.cast(attention_mask, dtype=dtype)


def make_causal_mask(mask: Tensor, dtype: str = "float32") -> Tensor:
  """Creates a 3D causal self-attention mask from a padding mask."""
  return keras.ops.tril(make_attention_mask(mask, dtype=dtype))


@keras.saving.register_keras_serializable("recml")
def gelu_approximate(x: Tensor) -> Tensor:
  """Approximate GELU activation function."""
  return keras.activations.gelu(x, approximate=True)


@keras.saving.register_keras_serializable("recml")
def relu_squared(x: Tensor) -> Tensor:
  """RELU squared activation function."""
  return keras.ops.square(keras.activations.relu(x))


@keras.saving.register_keras_serializable("recml")
def norm_embedding_post_processor(inputs: Tensor, eps: float = 1e-6) -> Tensor:
  """L2 Normalization Post Processor for HSTU.

  Take output embeddings and normalize them to unit length.

  Args:
    inputs: The input sequence tensor. shape = [B, N, D]
    eps: Epsilon to use for division.

  Returns:
    The normalized output embeddings.
  """
  return keras.ops.divide(
      inputs,
      keras.ops.clip(
          keras.ops.norm(inputs, ord=None, axis=-1, keepdims=True),
          x_min=eps,
          x_max=None,
      ),
  )


def apply_rotary_encoding(
    x: Tensor, *, positions: Tensor | None = None, max_wavelength: int
) -> Tensor:
  """Returns the rotary positional encodings.

  Args:
    x: Array of embeddings of shape [*batch_size, seq_len, num_heads, head_dim].
      Where head_dim must be even.
    positions: Optional array of shape [*batch_size, seq_len] holding the
      position of each token in the sequence. If not provided, the input is
      assumed to be a contiguous sequence and the positions are therefore [0, 1,
      ..., seq_len - 1] for each example.
    max_wavelength: Maximum wavelength that will appear in sin / cosine
      waveforms. This specifies the maximum sequence length for identifying
      unique positions.

  Returns:
    Array of rotary encoded input of shape [batch_size, seq_len, num_heads,
    head_dim].
  """
  x_shape = keras.ops.shape(x)
  b = (x_shape[i] for i in range(len(x_shape) - 3))
  seq_len = x_shape[-3]
  if x_shape[-1] % 2 != 0:
    raise ValueError(
        "Embedding dimension must be even, but got"
        f" {x_shape[-1]} for input of shape {x_shape}."
    )
  if len(x_shape) < 4:
    raise ValueError(
        f"Unexpected input shape: {x_shape}. Expected shape of rank 4 or"
        " greater."
    )
  if positions is None:
    positions = keras.ops.tile(
        keras.ops.arange(seq_len)[None, :],
        (*[d if d is not None else -1 for d in b], 1),
    )
  # Only do shape checks on not TF backends.
  if keras.backend.backend() != "tensorflow":
    if keras.ops.shape(positions) != x_shape[:-2]:
      raise ValueError(
          f"Positions must be of shape: {(x_shape[:-2])} but got shape:"
          f" {keras.ops.shape(positions)}."
      )
  freq_exponents = (2.0 / x_shape[-1]) * keras.ops.arange(
      x_shape[-1] // 2, dtype="float32"
  )
  timescale = max_wavelength**freq_exponents
  timescale = timescale[
      (*[None for _ in b], None, slice(None))
  ]  # timescale[None, None, :] when len(b) == 1
  radians = keras.ops.cast(positions[..., None], "float32") / timescale
  radians = radians[..., None, :]
  # radians.shape = [...,L,1,d=D/2]
  sin, cos = keras.ops.sin(radians), keras.ops.cos(radians)
  x1, x2 = keras.ops.split(x, 2, axis=-1)
  x1, x2 = keras.ops.cast(x1, "float32"), keras.ops.cast(x2, "float32")
  res = keras.ops.concatenate(
      [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
  )
  return keras.ops.cast(res, keras.ops.dtype(x))


def large_negative_for_attention(dtype: Any) -> float:
  """Return a large negative number based on dtype."""
  if keras.backend.standardize_dtype(dtype) == "float16":
    return -3e4
  return -1e9
