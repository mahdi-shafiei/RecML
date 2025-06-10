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
"""Pointwise attention tests."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from recml.core.ops import hstu_ops


class PointwiseAttentionTest(absltest.TestCase):

  def test_pointwise_attention(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Skipping TPU test.')

    batch_size, seq_len, num_heads, head_dim = 128, 512, 8, 128
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(42), 5)

    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim))
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim))

    rows = []
    for i in range(batch_size):
      l = int(jax.random.uniform(jax.random.fold_in(k4, i)) * seq_len)
      rows.append(np.concatenate((np.ones(l), np.zeros(seq_len - l))))
    segment_ids = jnp.asarray(np.stack(rows), jnp.int32)

    targets = jax.random.normal(k5, (batch_size, seq_len, num_heads, head_dim))

    @jax.jit
    def _pw_attn_ref(q, k, v):
      q = q * 1.0 / (q.shape[-1] ** 0.5)

      logits = jnp.einsum('btnh,bsnh->bnts', q, k)
      s = jax.nn.silu(logits) / q.shape[1]

      mask = jnp.tril(
          jnp.equal(segment_ids[..., :, None], segment_ids[..., None, :])
      )[:, None, :, :]
      s = jnp.where(mask, s, jnp.zeros_like(s))

      attn = jnp.einsum('bnts,bsnh->btnh', s, v)

      return jnp.sum((attn - targets) ** 2), attn

    @jax.jit
    def _pw_splash_attn(q, k, v):
      attn = hstu_ops.pointwise_splash_attention(
          hstu_ops.MaskType.CAUSAL,
          q,
          k,
          v,
          q_segment_ids=segment_ids,
          kv_segment_ids=segment_ids,
      )
      return jnp.sum((attn - targets) ** 2), attn

    (_, attn_ref), (dq_ref, dk_ref, dv_ref) = jax.value_and_grad(
        _pw_attn_ref, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    (_, attn), (dq, dk, dv) = jax.value_and_grad(
        _pw_splash_attn, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    np.testing.assert_allclose(attn, attn_ref, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(dv, dv_ref, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(dk, dk_ref, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(dq, dq_ref, atol=5e-3, rtol=5e-3)


if __name__ == '__main__':
  absltest.main()
