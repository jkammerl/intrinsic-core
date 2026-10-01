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

"""Loads models from a Triton model repository and runs them.

Supports the model repository layout of Triton
(`<repository>/<model name>/config.pbtxt` and
`<repository>/<model name>/<version>/<model file>`) and two backends:

- "onnxruntime": runs an ONNX model with ONNX Runtime.
- "python": runs a `model.py` that defines a Triton `TritonPythonModel`.
"""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import importlib.util
import json
import os
import sys
import threading

from absl import logging
from google.protobuf import json_format
from google.protobuf import text_format
import numpy as np
from tritonclient.grpc import model_config_pb2

from intrinsic_inference.onnxruntime_server import execution_providers
from intrinsic_inference.onnxruntime_server import triton_python_backend_utils

_CONFIG_FILENAME = "config.pbtxt"

_DEFAULT_MODEL_FILENAMES = {
    "onnxruntime": "model.onnx",
    "python": "model.py",
}

_PLATFORM_TO_BACKEND = {
    "onnxruntime_onnx": "onnxruntime",
}

_BACKEND_TO_PLATFORM = {
    "onnxruntime": "onnxruntime_onnx",
    "python": "python",
}

# Model config data types and the corresponding numpy dtypes.
_DATA_TYPE_TO_NUMPY = {
    model_config_pb2.TYPE_BOOL: np.dtype(np.bool_),
    model_config_pb2.TYPE_UINT8: np.dtype(np.uint8),
    model_config_pb2.TYPE_UINT16: np.dtype(np.uint16),
    model_config_pb2.TYPE_UINT32: np.dtype(np.uint32),
    model_config_pb2.TYPE_UINT64: np.dtype(np.uint64),
    model_config_pb2.TYPE_INT8: np.dtype(np.int8),
    model_config_pb2.TYPE_INT16: np.dtype(np.int16),
    model_config_pb2.TYPE_INT32: np.dtype(np.int32),
    model_config_pb2.TYPE_INT64: np.dtype(np.int64),
    model_config_pb2.TYPE_FP16: np.dtype(np.float16),
    model_config_pb2.TYPE_FP32: np.dtype(np.float32),
    model_config_pb2.TYPE_FP64: np.dtype(np.float64),
    model_config_pb2.TYPE_STRING: np.dtype(np.object_),
}

# ONNX Runtime tensor types and the corresponding model config data types.
_ORT_TYPE_TO_DATA_TYPE = {
    "tensor(bool)": model_config_pb2.TYPE_BOOL,
    "tensor(uint8)": model_config_pb2.TYPE_UINT8,
    "tensor(uint16)": model_config_pb2.TYPE_UINT16,
    "tensor(uint32)": model_config_pb2.TYPE_UINT32,
    "tensor(uint64)": model_config_pb2.TYPE_UINT64,
    "tensor(int8)": model_config_pb2.TYPE_INT8,
    "tensor(int16)": model_config_pb2.TYPE_INT16,
    "tensor(int32)": model_config_pb2.TYPE_INT32,
    "tensor(int64)": model_config_pb2.TYPE_INT64,
    "tensor(float16)": model_config_pb2.TYPE_FP16,
    "tensor(float)": model_config_pb2.TYPE_FP32,
    "tensor(double)": model_config_pb2.TYPE_FP64,
    "tensor(string)": model_config_pb2.TYPE_STRING,
}


class ErrorCode(enum.Enum):
  """Error categories, mapped to gRPC status codes by the server."""

  NOT_FOUND = "not_found"
  INVALID_ARGUMENT = "invalid_argument"
  INTERNAL = "internal"


class ModelError(Exception):
  """An error loading or running a model."""

  def __init__(self, code: ErrorCode, message: str) -> None:
    super().__init__(message)
    self.code = code


def numpy_dtype(data_type: int) -> np.dtype:
  """Returns the numpy dtype for a model config data type."""
  return _DATA_TYPE_TO_NUMPY[data_type]


def _dims_with_batch(
    config: model_config_pb2.ModelConfig, dims: Sequence[int]
) -> list[int]:
  """Returns the full tensor shape, including the batch dimension if any."""
  return ([-1] if config.max_batch_size > 0 else []) + list(dims)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
  name: str
  dtype: np.dtype
  shape: list[int]
  optional: bool = False


class Model(abc.ABC):
  """A loaded model version."""

  def __init__(
      self,
      config: model_config_pb2.ModelConfig,
      version: str,
      model_dir: str,
      device: execution_providers.Device,
  ) -> None:
    self.config = config
    self.version = version
    self.model_dir = model_dir
    self.device = device

  @property
  def name(self) -> str:
    return self.config.name

  @property
  def platform(self) -> str:
    return _BACKEND_TO_PLATFORM.get(self.config.backend, self.config.backend)

  @property
  def inputs(self) -> list[TensorSpec]:
    return [
        TensorSpec(
            name=i.name,
            dtype=numpy_dtype(i.data_type),
            shape=_dims_with_batch(self.config, i.dims),
            optional=i.optional,
        )
        for i in self.config.input
    ]

  @property
  def outputs(self) -> list[TensorSpec]:
    return [
        TensorSpec(
            name=o.name,
            dtype=numpy_dtype(o.data_type),
            shape=_dims_with_batch(self.config, o.dims),
        )
        for o in self.config.output
    ]

  def infer(
      self,
      inputs: Mapping[str, np.ndarray],
      requested_outputs: Sequence[str] = (),
  ) -> dict[str, np.ndarray]:
    """Runs inference.

    Args:
      inputs: Input tensors by name.
      requested_outputs: Names of the outputs to return, all if empty.

    Returns:
      The output tensors by name.

    Raises:
      ModelError: If the inputs are invalid or inference fails.
    """
    self._validate_inputs(inputs)
    known_outputs = {o.name for o in self.config.output}
    unknown = [n for n in requested_outputs if n not in known_outputs]
    if known_outputs and unknown:
      raise ModelError(
          ErrorCode.INVALID_ARGUMENT,
          f"unexpected inference output {unknown} for model '{self.name}'",
      )
    try:
      outputs = self._infer(inputs, requested_outputs)
    except ModelError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Inference failed for model '%s'", self.name)
      raise ModelError(
          ErrorCode.INTERNAL, f"inference failed for model '{self.name}': {e}"
      ) from e
    if requested_outputs:
      outputs = {n: outputs[n] for n in requested_outputs if n in outputs}
    return outputs

  def _validate_inputs(self, inputs: Mapping[str, np.ndarray]) -> None:
    specs = {s.name: s for s in self.inputs}
    for name in inputs:
      if specs and name not in specs:
        raise ModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"unexpected inference input '{name}' for model '{self.name}'",
        )
    for spec in specs.values():
      if spec.name not in inputs:
        if spec.optional:
          continue
        raise ModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"expected input '{spec.name}' for model '{self.name}'",
        )
      array = inputs[spec.name]
      if array.dtype != spec.dtype and not (
          spec.dtype == np.object_ and array.dtype.kind in "OSU"
      ):
        raise ModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"input '{spec.name}' of model '{self.name}' has data type"
            f" {array.dtype}, expected {spec.dtype}",
        )
      if len(array.shape) != len(spec.shape) or any(
          d != -1 and d != s for d, s in zip(spec.shape, array.shape)
      ):
        raise ModelError(
            ErrorCode.INVALID_ARGUMENT,
            f"input '{spec.name}' of model '{self.name}' has shape"
            f" {list(array.shape)}, expected {spec.shape}",
        )

  @abc.abstractmethod
  def _infer(
      self,
      inputs: Mapping[str, np.ndarray],
      requested_outputs: Sequence[str],
  ) -> dict[str, np.ndarray]:
    """Runs inference on validated inputs."""

  def unload(self) -> None:
    """Releases the model's resources."""


class OnnxRuntimeModel(Model):
  """Runs an ONNX model with ONNX Runtime."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    filename = self.config.default_model_filename or "model.onnx"
    path = os.path.join(self.model_dir, self.version, filename)
    if not os.path.exists(path):
      raise ModelError(
          ErrorCode.INTERNAL, f"model file '{path}' does not exist"
      )
    self._session = execution_providers.create_session(path, self.device)
    self._complete_config()

  @property
  def providers(self) -> list[str]:
    return self._session.get_providers()

  def _complete_config(self) -> None:
    """Fills in inputs and outputs missing from the config from the model."""
    if not self.config.input:
      for node in self._session.get_inputs():
        self.config.input.add(
            name=node.name,
            data_type=_ORT_TYPE_TO_DATA_TYPE[node.type],
            dims=[d if isinstance(d, int) else -1 for d in node.shape],
        )
    if not self.config.output:
      for node in self._session.get_outputs():
        self.config.output.add(
            name=node.name,
            data_type=_ORT_TYPE_TO_DATA_TYPE[node.type],
            dims=[d if isinstance(d, int) else -1 for d in node.shape],
        )

  def _infer(
      self,
      inputs: Mapping[str, np.ndarray],
      requested_outputs: Sequence[str],
  ) -> dict[str, np.ndarray]:
    output_names = list(requested_outputs) or [
        o.name for o in self._session.get_outputs()
    ]
    results = self._session.run(output_names, dict(inputs))
    return dict(zip(output_names, results))


class PythonModel(Model):
  """Runs a `model.py` written for Triton's Python backend."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    filename = self.config.default_model_filename or "model.py"
    version_dir = os.path.join(self.model_dir, self.version)
    path = os.path.join(version_dir, filename)
    if not os.path.exists(path):
      raise ModelError(
          ErrorCode.INTERNAL, f"model file '{path}' does not exist"
      )
    if "EXECUTION_ENV_PATH" in self.config.parameters:
      logging.info(
          "Ignoring EXECUTION_ENV_PATH of model '%s'; its dependencies must be"
          " installed in the server's environment.",
          self.name,
      )

    # Models import the Triton utilities under this name.
    sys.modules.setdefault(
        "triton_python_backend_utils", triton_python_backend_utils
    )
    if version_dir not in sys.path:
      sys.path.insert(0, version_dir)
    os.environ[execution_providers.DEVICE_ENV_VAR] = self.device.value

    module_name = f"_triton_python_model_{self.name}_{self.version}".replace(
        ".", "_"
    ).replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    self._model = module.TritonPythonModel()
    # Triton's Python backend runs one request batch at a time per instance.
    self._lock = threading.Lock()
    if hasattr(self._model, "initialize"):
      self._model.initialize(self._initialize_args())

  def _initialize_args(self) -> dict[str, str]:
    config = json_format.MessageToDict(
        self.config, preserving_proto_field_name=True
    )
    # MessageToDict renders int64 fields such as dims as strings.
    for key in ("input", "output"):
      for tensor in config.get(key, []):
        if "dims" in tensor:
          tensor["dims"] = [int(d) for d in tensor["dims"]]
    kind = "CPU" if self.device == execution_providers.Device.CPU else "GPU"
    return {
        "model_config": json.dumps(config),
        "model_instance_kind": kind,
        "model_instance_name": f"{self.name}_0",
        "model_instance_device_id": "0",
        "model_repository": self.model_dir,
        "model_version": self.version,
        "model_name": self.name,
    }

  def _infer(
      self,
      inputs: Mapping[str, np.ndarray],
      requested_outputs: Sequence[str],
  ) -> dict[str, np.ndarray]:
    pb_utils = triton_python_backend_utils
    request = pb_utils.InferenceRequest(
        inputs=[pb_utils.Tensor(n, a) for n, a in inputs.items()],
        requested_output_names=list(requested_outputs)
        or [o.name for o in self.config.output],
        model_name=self.name,
    )
    with self._lock:
      responses = self._model.execute([request])
    if not responses:
      raise ModelError(
          ErrorCode.INTERNAL, f"model '{self.name}' returned no response"
      )
    response = responses[0]
    if response.has_error():
      raise ModelError(ErrorCode.INTERNAL, response.error().message())
    return {t.name(): t.as_numpy() for t in response.output_tensors()}

  def unload(self) -> None:
    if hasattr(self._model, "finalize"):
      with self._lock:
        self._model.finalize()


_BACKENDS = {
    "onnxruntime": OnnxRuntimeModel,
    "python": PythonModel,
}


@dataclasses.dataclass(frozen=True)
class ModelStatus:
  name: str
  version: str
  state: str  # "READY" or "UNAVAILABLE", as in Triton's repository index.
  reason: str = ""


class ModelRepository:
  """Loads models from a Triton model repository directory."""

  def __init__(
      self, path: str, device: execution_providers.Device
  ) -> None:
    self._path = path
    self._device = device
    self._models: dict[str, Model] = {}
    self._failures: dict[str, str] = {}
    self._lock = threading.Lock()
    # Serializes loads and unloads, which can take long.
    self._load_lock = threading.Lock()

  @property
  def path(self) -> str:
    return self._path

  def get(self, name: str, version: str = "") -> Model:
    """Returns a loaded model.

    Raises:
      ModelError: If the model (version) is not loaded.
    """
    with self._lock:
      model = self._models.get(name)
    if model is None or (version and version != model.version):
      raise ModelError(
          ErrorCode.NOT_FOUND,
          f"Request for unknown model: '{name}'"
          + (f" version {version}" if version else "")
          + " is not found",
      )
    return model

  def is_ready(self, name: str, version: str = "") -> bool:
    try:
      self.get(name, version)
    except ModelError:
      return False
    return True

  def index(self) -> list[ModelStatus]:
    """Returns the status of all models in the repository."""
    names = set()
    if os.path.isdir(self._path):
      names = {
          e.name for e in os.scandir(self._path) if e.is_dir()
      }
    with self._lock:
      names |= set(self._models)
      statuses = []
      for name in sorted(names):
        model = self._models.get(name)
        if model is not None:
          statuses.append(ModelStatus(name, model.version, "READY"))
        else:
          statuses.append(
              ModelStatus(
                  name, "", "UNAVAILABLE", self._failures.get(name, "unloaded")
              )
          )
    return statuses

  def load(self, name: str, config_json: str | None = None) -> Model:
    """Loads (or reloads) a model.

    Args:
      name: The model's directory name in the repository.
      config_json: Optional model config in JSON, overriding config.pbtxt.

    Returns:
      The loaded model.

    Raises:
      ModelError: If loading fails.
    """
    with self._load_lock:
      try:
        model = self._create_model(name, config_json)
      except ModelError as e:
        with self._lock:
          self._failures[name] = str(e)
        raise ModelError(
            e.code, f"failed to load '{name}': {e}"
        ) from e
      except Exception as e:  # pylint: disable=broad-except
        logging.exception("Failed to load model '%s'", name)
        with self._lock:
          self._failures[name] = str(e)
        raise ModelError(
            ErrorCode.INTERNAL, f"failed to load '{name}': {e}"
        ) from e
      with self._lock:
        previous = self._models.get(name)
        self._models[name] = model
        self._failures.pop(name, None)
      if previous is not None:
        previous.unload()
      logging.info(
          "Loaded model '%s' version %s (%s)", name, model.version,
          model.config.backend,
      )
      return model

  def unload(self, name: str) -> None:
    """Unloads a model. Does nothing if it is not loaded."""
    with self._load_lock:
      with self._lock:
        model = self._models.pop(name, None)
      if model is not None:
        model.unload()
        logging.info("Unloaded model '%s'", name)

  def unload_all(self) -> None:
    with self._lock:
      names = list(self._models)
    for name in names:
      self.unload(name)

  def _create_model(self, name: str, config_json: str | None) -> Model:
    model_dir = os.path.join(self._path, name)
    if not os.path.isdir(model_dir):
      raise ModelError(
          ErrorCode.NOT_FOUND,
          f"model directory '{model_dir}' does not exist",
      )
    config = model_config_pb2.ModelConfig()
    if config_json:
      json_format.Parse(config_json, config)
    else:
      config_path = os.path.join(model_dir, _CONFIG_FILENAME)
      if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
          text_format.Parse(f.read(), config)
    if not config.name:
      config.name = name
    if config.name != name:
      raise ModelError(
          ErrorCode.INVALID_ARGUMENT,
          f"model name '{config.name}' in the config does not match the"
          f" directory name '{name}'",
      )

    backend = config.backend or _PLATFORM_TO_BACKEND.get(config.platform)
    if backend is None:
      backend = self._detect_backend(model_dir)
    if backend not in _BACKENDS:
      raise ModelError(
          ErrorCode.INVALID_ARGUMENT,
          f"backend '{backend or config.platform}' of model '{name}' is not"
          f" supported; supported backends: {sorted(_BACKENDS)}",
      )
    config.backend = backend

    version = self._latest_version(model_dir)
    return _BACKENDS[backend](config, version, model_dir, self._device)

  @staticmethod
  def _latest_version(model_dir: str) -> str:
    versions = [
        int(e.name)
        for e in os.scandir(model_dir)
        if e.is_dir() and e.name.isdigit()
    ]
    if not versions:
      raise ModelError(
          ErrorCode.NOT_FOUND,
          f"model directory '{model_dir}' has no version subdirectory",
      )
    return str(max(versions))

  @classmethod
  def _detect_backend(cls, model_dir: str) -> str | None:
    """Returns the backend implied by the model files, as Triton does."""
    version_dir = os.path.join(model_dir, cls._latest_version(model_dir))
    for backend, filename in _DEFAULT_MODEL_FILENAMES.items():
      if os.path.exists(os.path.join(version_dir, filename)):
        return backend
    return None

