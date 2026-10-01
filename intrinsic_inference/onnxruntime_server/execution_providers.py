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

"""Selects ONNX Runtime execution providers based on the available hardware."""

from collections.abc import Sequence
import ctypes
import enum
import functools
import glob
import os

from absl import logging
import onnxruntime as ort

# Environment variable through which the selected device is passed on to
# Python backend models (e.g. FoundationPose's model.py).
DEVICE_ENV_VAR = "INTRINSIC_INFERENCE_DEVICE"

CPU_PROVIDER = "CPUExecutionProvider"

# GPU execution providers in order of preference.
GPU_PROVIDERS = ("CUDAExecutionProvider", "CoreMLExecutionProvider")


# CUDA and cuDNN libraries of the nvidia-* wheels needed by the CUDA execution
# provider, in dependency order. The cuDNN sub-libraries are loaded by cuDNN
# on demand, by soname.
_CUDA_LIBRARIES = (
    "libcudart.so.*",
    "libnvJitLink.so.*",
    "libcublasLt.so.*",
    "libcublas.so.*",
    "libnvrtc.so.*",
    "libcurand.so.*",
    "libcufft.so.*",
    "libcudnn_graph.so.*",
    "libcudnn_engines_precompiled.so.*",
    "libcudnn_engines_runtime_compiled.so.*",
    "libcudnn_heuristic.so.*",
    "libcudnn_ops.so.*",
    "libcudnn_cnn.so.*",
    "libcudnn_adv.so.*",
    "libcudnn.so.*",
)


@functools.cache
def cuda_driver_available() -> bool:
  """Returns whether the NVIDIA driver's CUDA library can be loaded."""
  try:
    ctypes.CDLL("libcuda.so.1")
  except OSError:
    return False
  return True


@functools.cache
def preload_cuda_libraries() -> list[str]:
  """Loads the CUDA and cuDNN libraries of the installed nvidia-* wheels.

  `onnxruntime.preload_dlls()` only looks for the wheels next to onnxruntime in
  the same site-packages directory. Bazel installs every wheel in its own
  directory, so the libraries are looked up through the `nvidia` namespace
  package instead. Libraries that can't be found are left to the dynamic
  loader, e.g. when CUDA is installed on the system.

  Returns:
    The paths of the loaded libraries.
  """
  try:
    import nvidia  # pylint: disable=g-import-not-at-top
  except ImportError:
    return []
  lib_dirs = []
  for package_dir in nvidia.__path__:
    lib_dirs += sorted(glob.glob(os.path.join(package_dir, "*", "lib")))
  loaded = []
  for pattern in _CUDA_LIBRARIES:
    for lib_dir in lib_dirs:
      paths = sorted(glob.glob(os.path.join(lib_dir, pattern)))
      if paths:
        try:
          ctypes.CDLL(paths[0], mode=ctypes.RTLD_GLOBAL)
          loaded.append(paths[0])
        except OSError as e:
          logging.warning("Failed to load %s: %s", paths[0], e)
        break
  logging.info("Preloaded %d CUDA libraries", len(loaded))
  return loaded


@enum.unique
class Device(enum.Enum):
  """Where to run inference."""

  # Use a GPU execution provider if one is available, otherwise the CPU.
  AUTO = "auto"
  CPU = "cpu"
  # Fail if no GPU execution provider can be used.
  GPU = "gpu"


def select_providers(
    device: Device, available: Sequence[str] | None = None
) -> list[str]:
  """Returns the execution providers to try, in order of preference.

  Args:
    device: Where to run inference.
    available: The providers available in this ONNX Runtime build. Defaults to
      `onnxruntime.get_available_providers()`.

  Raises:
    RuntimeError: If `device` is GPU but no GPU provider is available.
  """
  if available is None:
    available = ort.get_available_providers()
  if device == Device.CPU:
    return [CPU_PROVIDER]
  gpu = [p for p in GPU_PROVIDERS if p in available]
  if device == Device.GPU and not gpu:
    raise RuntimeError(
        "GPU inference was requested, but this ONNX Runtime build has no GPU"
        f" execution provider. Available providers: {list(available)}"
    )
  return gpu + [CPU_PROVIDER]


def is_gpu_session(session: ort.InferenceSession) -> bool:
  """Returns whether `session` runs on a GPU execution provider."""
  providers = session.get_providers()
  return bool(providers) and providers[0] in GPU_PROVIDERS


def create_session(
    model: str | bytes,
    device: Device,
    session_options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
  """Creates an inference session on the best available device.

  ONNX Runtime falls back to the CPU if a GPU provider fails to initialize
  (e.g. when onnxruntime-gpu is installed but no GPU or driver is present), so
  the providers actually in use are checked after the session is created.

  Args:
    model: Path to, or serialized bytes of, an ONNX model.
    device: Where to run inference.
    session_options: Optional session options.

  Returns:
    The inference session.

  Raises:
    RuntimeError: If `device` is GPU but the session could not use a GPU.
  """
  providers = select_providers(device)
  if "CUDAExecutionProvider" in providers:
    if device == Device.AUTO and not cuda_driver_available():
      # Without a driver the CUDA provider can't initialize. Skip it instead of
      # loading CUDA and letting ONNX Runtime log an error and fall back.
      providers.remove("CUDAExecutionProvider")
    else:
      preload_cuda_libraries()
  session = ort.InferenceSession(
      model, sess_options=session_options, providers=providers
  )
  if device == Device.GPU and not is_gpu_session(session):
    raise RuntimeError(
        "GPU inference was requested, but ONNX Runtime could not initialize a"
        f" GPU execution provider (using {session.get_providers()})."
    )
  logging.info("Created ONNX Runtime session with %s", session.get_providers())
  return session
