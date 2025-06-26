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
"""Sparsecore tests."""

import functools

from absl.testing import absltest
from etils import epy
import jax
import numpy as np
from recml.core.training import partitioning
from recml.layers.linen import sparsecore
import tensorflow as tf

with epy.lazy_imports():
  from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec  # pylint: disable=g-import-not-at-top


class SparsecoreTest(absltest.TestCase):

  def test_sparsecore_embedder_equivalence(self):
    if jax.devices()[0].platform != "tpu":
      self.skipTest("Test only supported on TPUs.")

    tf.random.set_seed(0)

    inputs = {
        "a": tf.random.uniform(
            shape=(32, 16), minval=1, maxval=100, dtype=tf.int64
        ),
        "b": tf.random.uniform(
            shape=(32, 16), minval=1, maxval=100, dtype=tf.int64
        ),
        "w": tf.random.normal(shape=(32, 16), dtype=tf.float32),
    }

    dp_partitioner = partitioning.DataParallelPartitioner()
    sparsecore_config = sparsecore.SparsecoreConfig(
        specs={
            "a": sparsecore.EmbeddingSpec(
                input_dim=100,
                embedding_dim=64,
                combiner="mean",
                weight_name="w",
            ),
            "b": sparsecore.EmbeddingSpec(
                input_dim=100,
                embedding_dim=64,
                max_sequence_length=16,
            ),
        },
        optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
    )
    preprocessor = sparsecore.SparsecorePreprocessor(sparsecore_config, 32)
    layer = sparsecore.SparsecoreEmbed(sparsecore_config)

    sc_inputs = dp_partitioner.shard_inputs(preprocessor(inputs))
    sc_vars = dp_partitioner.partition_init(
        functools.partial(layer.init, jax.random.key(0)),
        abstract_batch=sc_inputs,
    )(sc_inputs)

    def step(inputs, params):
      return layer.apply(params, inputs)

    p_step = dp_partitioner.partition_step(step, training=False)
    sparsecore_activations = jax.device_get(p_step(sc_inputs, sc_vars))

    self.assertEqual(sparsecore_activations["a"].shape, (32, 64))
    self.assertEqual(sparsecore_activations["b"].shape, (32, 16, 64))

    tables = sparsecore.fetch_tables(
        sparsecore_config,
        sc_vars["params"][sparsecore.EMBEDDING_PARAM_NAME],
        donate=False,
    )

    activations = sparsecore.cpu_lookup(sparsecore_config, tables, inputs)
    np.testing.assert_allclose(
        sparsecore_activations["a"], activations["a"], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        sparsecore_activations["b"], activations["b"], rtol=1e-5, atol=1e-5
    )

    np.testing.assert_allclose(
        sparsecore.gather_table(
            sparsecore_config,
            sc_vars["params"][sparsecore.EMBEDDING_PARAM_NAME],
            "a",
        ),
        tables["a"],
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
  absltest.main()
