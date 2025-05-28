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
"""Tests for layer utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import keras
from keras.src import testing
import numpy as np
from recml.layers.keras import utils


class UtilsTest(testing.TestCase):

  def test_clone_initializer(self):
    random_initializer = keras.initializers.RandomNormal(stddev=1.0)
    random_initializer_clone = utils.clone_initializer(random_initializer)
    self.assertNotEqual(random_initializer.seed, random_initializer_clone.seed)

    lecun_initializer = keras.initializers.LecunNormal(seed=1)
    lecun_initializer_clone = utils.clone_initializer(lecun_initializer)
    self.assertEqual(lecun_initializer.seed, lecun_initializer_clone.seed)

    self.assertEqual(utils.clone_initializer("lecun_normal"), "lecun_normal")

  # Remember to read these sideways =))
  @parameterized.parameters(
      dict(
          inputs=np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
          expected_outputs=np.array(
              [
                  [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
              ],
              dtype=np.float32,
          ),
      ),
  )
  def test_make_attention_mask(
      self, inputs: np.array, expected_outputs: np.array
  ):
    self.assertAllClose(
        utils.make_attention_mask(keras.ops.array(inputs)),
        keras.ops.array(expected_outputs),
    )

  @parameterized.parameters(
      dict(
          inputs=np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
          expected_outputs=np.array(
              [
                  [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
              ],
              dtype=np.float32,
          ),
      ),
  )
  def test_make_causal_mask(self, inputs: np.array, expected_outputs: np.array):
    self.assertAllClose(
        utils.make_causal_mask(keras.ops.array(inputs)),
        keras.ops.array(expected_outputs),
    )

  @parameterized.parameters(
      dict(
          inputs=np.array([[[1.0, 1.0], [0.0, 1.0]]]),
          eps=1e-6,
          expected_output=np.array([[[0.70710678, 0.70710678], [0.0, 1.0]]]),
      ),
      dict(
          inputs=np.array([[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]]),
          eps=1e-06,
          expected_output=np.array(
              [[[0.0, 1.0, 0.0], [0.5773502, 0.5773502, 0.5773502]]]
          ),
      ),
  )
  def test_l2_norm_embedding_postprocessor_output(
      self, inputs, eps, expected_output
  ):
    inputs = keras.ops.array(inputs)
    expected_output = keras.ops.array(expected_output)

    got = utils.norm_embedding_post_processor(inputs, eps=eps)
    self.assertAllClose(got, expected_output)

  @parameterized.named_parameters(
      dict(
          # wavelength == 1 so radians == expand_dims(positions, axes=[-1, -2])
          testcase_name="with_positions",
          inputs=np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]]),
          positions=np.array([[4, 1]]),
          max_wavelength=1,
          expected_outputs=np.array([[
              [[
                  1.0 * np.cos(4) - 2.0 * np.sin(4),
                  2.0 * np.cos(4) + 1.0 * np.sin(4),
              ]],
              [[
                  3.0 * np.cos(1) - 4.0 * np.sin(1),
                  4.0 * np.cos(1) + 3.0 * np.sin(1),
              ]],
          ]]),
      ),
      dict(
          # wavelength == 1 so radians == expand_dims(positions, axes=[-1, -2])
          testcase_name="no_positions",
          inputs=np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]]),
          positions=None,  # Should evaluate to [[0, 1]]
          max_wavelength=1,
          expected_outputs=np.array([[
              [[1.0, 2.0]],
              [[
                  3.0 * np.cos(1) - 4.0 * np.sin(1),
                  4.0 * np.cos(1) + 3.0 * np.sin(1),
              ]],
          ]]),
      ),
  )
  def test_apply_rotary_embedding(
      self,
      inputs: np.ndarray,
      positions: np.ndarray | None,
      max_wavelength: int,
      expected_outputs: np.ndarray,
  ):
    self.assertAllClose(
        expected_outputs,
        utils.apply_rotary_encoding(
            inputs,
            positions=positions,
            max_wavelength=max_wavelength,
        ),
    )


if __name__ == "__main__":
  absltest.main()
