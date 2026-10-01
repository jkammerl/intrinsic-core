# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from absl.testing import absltest
import numpy as np

from intrinsic_inference.onnxruntime_server import execution_providers
from intrinsic_inference.onnxruntime_server import models
from intrinsic_inference.onnxruntime_server import testing_models

Device = execution_providers.Device


def _segmentation_inputs(h=4, w=5, scale=2.0, threshold=5.0):
  image = np.arange(3 * h * w, dtype=np.uint8).reshape(1, 3, h, w)
  thresholds = np.array([[scale, threshold]], dtype=np.float32)
  return {"input": image, "thresholds": thresholds}


class OnnxRuntimeModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.repo_path = self.create_tempdir().full_path
    self.repo = models.ModelRepository(self.repo_path, Device.CPU)

  def test_infer_with_dynamic_shapes(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    for h, w in [(4, 5), (7, 3)]:
      inputs = _segmentation_inputs(h, w)
      outputs = model.infer(inputs)
      expected = inputs["input"].astype(np.float32) * 2.0
      np.testing.assert_array_equal(outputs["mean"], expected)
      np.testing.assert_array_equal(outputs["mask"], expected > 5.0)
      self.assertEqual(outputs["mask"].dtype, np.bool_)

  def test_infer_returns_only_requested_outputs(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    outputs = model.infer(_segmentation_inputs(), ["mask"])
    self.assertEqual(list(outputs), ["mask"])

  def test_model_properties(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    self.assertEqual(model.platform, "onnxruntime_onnx")
    self.assertEqual(model.version, "1")
    self.assertEqual(model.providers, ["CPUExecutionProvider"])
    self.assertEqual(
        [(s.name, s.dtype, s.shape) for s in model.inputs],
        [
            ("input", np.uint8, [1, 3, -1, -1]),
            ("thresholds", np.float32, [1, 2]),
        ],
    )

  def test_config_is_completed_from_model_without_config_pbtxt(self):
    testing_models.write_segmentation_like_model(
        self.repo_path, "seg", with_config=False
    )
    model = self.repo.load("seg")
    self.assertEqual(model.config.backend, "onnxruntime")
    self.assertEqual(
        [(s.name, s.dtype, s.shape) for s in model.outputs],
        [
            ("mean", np.float32, [1, 3, -1, -1]),
            ("mask", np.bool_, [1, 3, -1, -1]),
        ],
    )
    outputs = model.infer(_segmentation_inputs())
    self.assertEqual(outputs["mean"].shape, (1, 3, 4, 5))

  def test_loads_latest_version(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg", "1")
    testing_models.write_segmentation_like_model(self.repo_path, "seg", "10")
    testing_models.write_segmentation_like_model(self.repo_path, "seg", "2")
    self.assertEqual(self.repo.load("seg").version, "10")
    self.assertTrue(self.repo.is_ready("seg", "10"))
    self.assertFalse(self.repo.is_ready("seg", "2"))

  def test_wrong_dtype_is_invalid_argument(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    inputs = _segmentation_inputs()
    inputs["input"] = inputs["input"].astype(np.float32)
    with self.assertRaisesRegex(models.ModelError, "data type") as e:
      model.infer(inputs)
    self.assertEqual(e.exception.code, models.ErrorCode.INVALID_ARGUMENT)

  def test_wrong_shape_is_invalid_argument(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    inputs = _segmentation_inputs()
    inputs["thresholds"] = np.zeros((1, 3), np.float32)
    with self.assertRaisesRegex(models.ModelError, "shape") as e:
      model.infer(inputs)
    self.assertEqual(e.exception.code, models.ErrorCode.INVALID_ARGUMENT)

  def test_missing_input_is_invalid_argument(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    inputs = _segmentation_inputs()
    del inputs["thresholds"]
    with self.assertRaisesRegex(models.ModelError, "expected input") as e:
      model.infer(inputs)
    self.assertEqual(e.exception.code, models.ErrorCode.INVALID_ARGUMENT)

  def test_unknown_output_is_invalid_argument(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    model = self.repo.load("seg")
    with self.assertRaisesRegex(models.ModelError, "unexpected inference output"):
      model.infer(_segmentation_inputs(), ["nope"])

  def test_config_json_overrides_config_pbtxt(self):
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    config = {
        "name": "seg",
        "backend": "onnxruntime",
        "default_model_filename": "model.onnx",
    }
    model = self.repo.load("seg", json.dumps(config))
    # The config has no inputs, so they are completed from the model.
    self.assertLen(model.config.input, 2)
    self.assertEmpty(model.config.instance_group)


class PythonModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.repo_path = self.create_tempdir().full_path
    self.model_dir = testing_models.write_python_model(self.repo_path, "py")

  def test_initialize_and_infer(self):
    repo = models.ModelRepository(self.repo_path, Device.CPU)
    model = repo.load("py")
    self.assertTrue(os.path.exists(os.path.join(self.model_dir, "initialized")))
    self.assertEqual(model.platform, "python")

    outputs = model.infer(
        {
            "VALUES": np.array([1, 2, 3.5], np.float32),
            "SCALE": np.array([2], np.int32),
        }
    )
    np.testing.assert_array_equal(outputs["SUM"], [13.0])
    # The server's device setting is passed on to the model.
    self.assertEqual(outputs["DEVICE"].tolist(), [b"cpu"])

  def test_optional_input_may_be_omitted(self):
    repo = models.ModelRepository(self.repo_path, Device.AUTO)
    model = repo.load("py")
    outputs = model.infer({"VALUES": np.array([1, 2], np.float32)}, ["SUM"])
    self.assertEqual(list(outputs), ["SUM"])
    np.testing.assert_array_equal(outputs["SUM"], [3.0])

  def test_model_error_is_internal(self):
    repo = models.ModelRepository(self.repo_path, Device.CPU)
    model = repo.load("py")
    with self.assertRaisesRegex(models.ModelError, "negative values") as e:
      model.infer({"VALUES": np.array([-1], np.float32)})
    self.assertEqual(e.exception.code, models.ErrorCode.INTERNAL)

  def test_initialize_args(self):
    repo = models.ModelRepository(self.repo_path, Device.CPU)
    model = repo.load("py")
    args = model._model.args
    self.assertEqual(args["model_name"], "py")
    self.assertEqual(args["model_version"], "1")
    self.assertEqual(args["model_instance_kind"], "CPU")
    self.assertEqual(args["model_repository"], self.model_dir)
    config = json.loads(args["model_config"])
    self.assertEqual(config["input"][0]["dims"], [-1])
    self.assertEqual(config["output"][1]["data_type"], "TYPE_STRING")

  def test_unload_calls_finalize(self):
    repo = models.ModelRepository(self.repo_path, Device.CPU)
    repo.load("py")
    repo.unload("py")
    self.assertTrue(os.path.exists(os.path.join(self.model_dir, "finalized")))
    self.assertFalse(repo.is_ready("py"))


class ModelRepositoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.repo_path = self.create_tempdir().full_path
    self.repo = models.ModelRepository(self.repo_path, Device.CPU)

  def test_get_unknown_model_is_not_found(self):
    with self.assertRaisesRegex(models.ModelError, "unknown model") as e:
      self.repo.get("nope")
    self.assertEqual(e.exception.code, models.ErrorCode.NOT_FOUND)

  def test_load_missing_model_fails(self):
    with self.assertRaisesRegex(models.ModelError, "failed to load 'nope'"):
      self.repo.load("nope")

  def test_index_reports_state_and_failures(self):
    testing_models.write_segmentation_like_model(self.repo_path, "a")
    broken = testing_models.write_segmentation_like_model(self.repo_path, "b")
    os.remove(os.path.join(broken, "1", "model.onnx"))
    self.repo.load("a")
    with self.assertRaises(models.ModelError):
      self.repo.load("b")
    statuses = {s.name: s for s in self.repo.index()}
    self.assertEqual(statuses["a"].state, "READY")
    self.assertEqual(statuses["a"].version, "1")
    self.assertEqual(statuses["b"].state, "UNAVAILABLE")
    self.assertIn("does not exist", statuses["b"].reason)

  def test_reload_replaces_model(self):
    testing_models.write_python_model(self.repo_path, "py")
    first = self.repo.load("py")
    second = self.repo.load("py")
    self.assertIsNot(first, second)
    self.assertIs(self.repo.get("py"), second)

  def test_mismatched_config_name_fails(self):
    model_dir = testing_models.write_segmentation_like_model(
        self.repo_path, "seg"
    )
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
      f.write(testing_models.SEGMENTATION_LIKE_CONFIG.format(name="other"))
    with self.assertRaisesRegex(models.ModelError, "does not match"):
      self.repo.load("seg")

  def test_unsupported_backend_fails(self):
    model_dir = os.path.join(self.repo_path, "tf")
    os.makedirs(os.path.join(model_dir, "1"))
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
      f.write('name: "tf" platform: "tensorflow_savedmodel"')
    with self.assertRaisesRegex(models.ModelError, "not supported"):
      self.repo.load("tf")


if __name__ == "__main__":
  absltest.main()
