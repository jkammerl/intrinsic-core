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

"""In-process implementation of Triton's `triton_python_backend_utils`.

Python backend models (a `model.py` defining `TritonPythonModel`) import this
module as `triton_python_backend_utils`. This implements the subset of its API
that such models use, so that they run unchanged in the ONNX Runtime server.
"""

from collections.abc import Sequence
import json
from typing import Any

import numpy as np


class TritonError(Exception):
  """An error reported by a model."""

  def __init__(self, message: str) -> None:
    super().__init__(message)
    self._message = message

  def message(self) -> str:
    return self._message


class TritonModelException(Exception):
  """Raised by models to report an error."""


class Tensor:
  """A named tensor."""

  def __init__(self, name: str, array: np.ndarray) -> None:
    self._name = name
    self._array = np.asarray(array)

  def name(self) -> str:
    return self._name

  def as_numpy(self) -> np.ndarray:
    return self._array

  def is_cpu(self) -> bool:
    return True


class InferenceRequest:
  """An inference request passed to `TritonPythonModel.execute`."""

  def __init__(
      self,
      inputs: Sequence[Tensor],
      requested_output_names: Sequence[str] = (),
      request_id: str = "",
      model_name: str = "",
      parameters: dict[str, Any] | None = None,
  ) -> None:
    self._inputs = list(inputs)
    self._requested_output_names = list(requested_output_names)
    self._request_id = request_id
    self._model_name = model_name
    self._parameters = parameters or {}

  def inputs(self) -> list[Tensor]:
    return self._inputs

  def requested_output_names(self) -> list[str]:
    return self._requested_output_names

  def request_id(self) -> str:
    return self._request_id

  def model_name(self) -> str:
    return self._model_name

  def parameters(self) -> str:
    return json.dumps(self._parameters)


class InferenceResponse:
  """An inference response returned by `TritonPythonModel.execute`."""

  def __init__(
      self,
      output_tensors: Sequence[Tensor] = (),
      error: TritonError | None = None,
  ) -> None:
    self._output_tensors = list(output_tensors)
    self._error = error

  def output_tensors(self) -> list[Tensor]:
    return self._output_tensors

  def has_error(self) -> bool:
    return self._error is not None

  def error(self) -> TritonError | None:
    return self._error


def get_input_tensor_by_name(
    request: InferenceRequest, name: str
) -> Tensor | None:
  """Returns the input tensor `name` of `request`, or None if absent."""
  for tensor in request.inputs():
    if tensor.name() == name:
      return tensor
  return None


def get_output_tensor_by_name(
    response: InferenceResponse, name: str
) -> Tensor | None:
  """Returns the output tensor `name` of `response`, or None if absent."""
  for tensor in response.output_tensors():
    if tensor.name() == name:
      return tensor
  return None


def _get_config_by_name(
    model_config: dict[str, Any], key: str, name: str
) -> dict[str, Any]:
  for entry in model_config.get(key, []):
    if entry.get("name") == name:
      return entry
  return {}


def get_input_config_by_name(
    model_config: dict[str, Any], name: str
) -> dict[str, Any]:
  """Returns the config of input `name` from a model config dict."""
  return _get_config_by_name(model_config, "input", name)


def get_output_config_by_name(
    model_config: dict[str, Any], name: str
) -> dict[str, Any]:
  """Returns the config of output `name` from a model config dict."""
  return _get_config_by_name(model_config, "output", name)


_TRITON_TO_NUMPY = {
    "TYPE_BOOL": np.bool_,
    "TYPE_UINT8": np.uint8,
    "TYPE_UINT16": np.uint16,
    "TYPE_UINT32": np.uint32,
    "TYPE_UINT64": np.uint64,
    "TYPE_INT8": np.int8,
    "TYPE_INT16": np.int16,
    "TYPE_INT32": np.int32,
    "TYPE_INT64": np.int64,
    "TYPE_FP16": np.float16,
    "TYPE_FP32": np.float32,
    "TYPE_FP64": np.float64,
    "TYPE_STRING": np.object_,
}


def triton_string_to_numpy(triton_type: str) -> type[Any]:
  """Returns the numpy dtype for a Triton config data type such as TYPE_FP32."""
  return _TRITON_TO_NUMPY[triton_type]
