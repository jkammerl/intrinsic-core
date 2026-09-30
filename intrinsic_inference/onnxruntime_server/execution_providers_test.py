# Copyright 2026 Intrinsic Innovation LLC
#
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

import ctypes
import ctypes.util
import os
import shutil
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import onnxruntime as ort

from intrinsic_inference.onnxruntime_server import execution_providers
from intrinsic_inference.onnxruntime_server import testing_models

Device = execution_providers.Device
CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"


class SelectProvidersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("auto_cpu_only", Device.AUTO, [CPU], [CPU]),
      ("auto_cuda", Device.AUTO, [CUDA, CPU], [CUDA, CPU]),
      ("auto_coreml", Device.AUTO, [COREML, CPU], [COREML, CPU]),
      ("auto_ignores_other", Device.AUTO, ["AzureExecutionProvider", CPU], [CPU]),
      ("cpu_ignores_cuda", Device.CPU, [CUDA, CPU], [CPU]),
      ("gpu_cuda", Device.GPU, [CUDA, CPU], [CUDA, CPU]),
      ("prefers_cuda", Device.AUTO, [COREML, CUDA, CPU], [CUDA, COREML, CPU]),
  )
  def test_select_providers(self, device, available, expected):
    self.assertEqual(
        execution_providers.select_providers(device, available), expected
    )

  def test_gpu_without_gpu_provider_raises(self):
    with self.assertRaisesRegex(RuntimeError, "no GPU execution provider"):
      execution_providers.select_providers(Device.GPU, [CPU])


class CreateSessionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    repo = self.create_tempdir().full_path
    model_dir = testing_models.write_segmentation_like_model(repo, "m")
    self.model_path = os.path.join(model_dir, "1", "model.onnx")

  def test_cpu_session(self):
    session = execution_providers.create_session(self.model_path, Device.CPU)
    self.assertEqual(session.get_providers(), [CPU])
    self.assertFalse(execution_providers.is_gpu_session(session))

  def test_auto_session_uses_available_device(self):
    session = execution_providers.create_session(self.model_path, Device.AUTO)
    gpu_available = any(
        p in ort.get_available_providers()
        for p in execution_providers.GPU_PROVIDERS
    )
    if not gpu_available:
      self.assertEqual(session.get_providers(), [CPU])
    self.assertIn(CPU, session.get_providers())

  def test_gpu_session_fails_when_gpu_provider_falls_back(self):
    # Simulates onnxruntime-gpu on a machine without a usable GPU: the CUDA
    # provider is listed, but the session silently runs on the CPU.
    with mock.patch.object(
        ort, "get_available_providers", return_value=[CUDA, CPU]
    ), mock.patch.object(
        ort, "InferenceSession", autospec=True
    ) as session_cls:
      session_cls.return_value.get_providers.return_value = [CPU]
      with self.assertRaisesRegex(RuntimeError, "could not initialize a GPU"):
        execution_providers.create_session(self.model_path, Device.GPU)
      session_cls.assert_called_once_with(
          self.model_path, sess_options=None, providers=[CUDA, CPU]
      )


class PreloadCudaLibrariesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    execution_providers.preload_cuda_libraries.cache_clear()
    self.addCleanup(execution_providers.preload_cuda_libraries.cache_clear)

  def test_loads_libraries_from_all_nvidia_package_dirs(self):
    # Bazel installs each nvidia-* wheel in its own directory, which the
    # `nvidia` namespace package spans. A system library stands in for CUDA.
    ctypes.CDLL(ctypes.util.find_library("z"))
    with open("/proc/self/maps") as f:
      libz = next(
          line.split()[-1] for line in f if "/libz.so" in line.split()[-1]
      )
    cuda = self.create_tempdir().mkdir("nvidia").mkdir("cu13").mkdir("lib")
    cudnn = self.create_tempdir().mkdir("nvidia").mkdir("cudnn").mkdir("lib")
    shutil.copy(libz, os.path.join(cuda.full_path, "libcudart.so.13"))
    shutil.copy(libz, os.path.join(cuda.full_path, "libcublas.so.13"))
    shutil.copy(libz, os.path.join(cudnn.full_path, "libcudnn.so.9"))
    nvidia = types.ModuleType("nvidia")
    nvidia.__path__ = [
        os.path.dirname(os.path.dirname(cuda.full_path)),
        os.path.dirname(os.path.dirname(cudnn.full_path)),
    ]
    with mock.patch.dict(sys.modules, {"nvidia": nvidia}):
      loaded = execution_providers.preload_cuda_libraries()
    self.assertEqual(
        [os.path.basename(p) for p in loaded],
        ["libcudart.so.13", "libcublas.so.13", "libcudnn.so.9"],
    )

  def test_without_nvidia_wheels_loads_nothing(self):
    with mock.patch.dict(sys.modules, {"nvidia": None}):
      self.assertEmpty(execution_providers.preload_cuda_libraries())


if __name__ == "__main__":
  absltest.main()
