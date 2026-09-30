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

"""KServe v2 / Triton compatible gRPC inference service on ONNX Runtime.

Implements the parts of Triton's GRPCInferenceService that the inference
service uses (health, metadata, inference, explicit model control and system
shared memory), so that it can replace Triton on machines without an NVIDIA
GPU.
"""

from collections.abc import Callable
import functools
from typing import Any

from absl import logging
import grpc
import numpy as np
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tritonclient import utils as triton_utils

from intrinsic_inference.onnxruntime_server import models
from intrinsic_inference.onnxruntime_server import shared_memory

SERVER_NAME = "intrinsic-onnxruntime-server"
SERVER_VERSION = "1.0.0"

_EXTENSIONS = (
    "model_repository",
    "model_repository(unload_dependents)",
    "system_shared_memory",
)

_SHM_REGION = "shared_memory_region"
_SHM_OFFSET = "shared_memory_offset"
_SHM_BYTE_SIZE = "shared_memory_byte_size"

_ERROR_CODES = {
    models.ErrorCode.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    models.ErrorCode.INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    models.ErrorCode.INTERNAL: grpc.StatusCode.INTERNAL,
}

# Fields of InferTensorContents by datatype.
_CONTENTS_FIELDS = {
    "BOOL": "bool_contents",
    "INT8": "int_contents",
    "INT16": "int_contents",
    "INT32": "int_contents",
    "INT64": "int64_contents",
    "UINT8": "uint_contents",
    "UINT16": "uint_contents",
    "UINT32": "uint_contents",
    "UINT64": "uint64_contents",
    "FP32": "fp32_contents",
    "FP64": "fp64_contents",
    "BYTES": "bytes_contents",
}


class _RequestError(Exception):
  """An error to return to the client with the given status code."""

  def __init__(self, code: grpc.StatusCode, message: str) -> None:
    super().__init__(message)
    self.code = code


def _handle_errors(method: Callable[..., Any]) -> Callable[..., Any]:
  """Converts exceptions raised by `method` into gRPC errors."""

  @functools.wraps(method)
  def wrapper(self, request, context):
    try:
      return method(self, request, context)
    except _RequestError as e:
      context.abort(e.code, str(e))
    except models.ModelError as e:
      context.abort(_ERROR_CODES[e.code], str(e))
    except shared_memory.SharedMemoryError as e:
      context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Error handling %s", method.__name__)
      context.abort(grpc.StatusCode.INTERNAL, str(e))

  return wrapper


def _param_value(param: service_pb2.InferParameter) -> Any:
  return getattr(param, param.WhichOneof("parameter_choice"))


def _shm_params(parameters: Any) -> tuple[str, int, int] | None:
  """Returns (region, offset, byte size) if `parameters` use shared memory."""
  if _SHM_REGION not in parameters:
    return None
  region = _param_value(parameters[_SHM_REGION])
  byte_size = (
      int(_param_value(parameters[_SHM_BYTE_SIZE]))
      if _SHM_BYTE_SIZE in parameters
      else 0
  )
  offset = (
      int(_param_value(parameters[_SHM_OFFSET]))
      if _SHM_OFFSET in parameters
      else 0
  )
  if byte_size <= 0:
    raise _RequestError(
        grpc.StatusCode.INVALID_ARGUMENT,
        f"'{_SHM_BYTE_SIZE}' must be set for shared memory region '{region}'",
    )
  return region, offset, byte_size


def _np_dtype(datatype: str) -> np.dtype:
  dtype = triton_utils.triton_to_np_dtype(datatype)
  if dtype is None:
    raise _RequestError(
        grpc.StatusCode.INVALID_ARGUMENT, f"unsupported datatype '{datatype}'"
    )
  return np.dtype(dtype)


def decode_tensor(
    tensor: service_pb2.ModelInferRequest.InferInputTensor,
    raw: bytes | None,
) -> np.ndarray:
  """Converts an input tensor from its raw bytes or typed contents."""
  shape = list(tensor.shape)
  dtype = _np_dtype(tensor.datatype)
  try:
    if raw is not None:
      if tensor.datatype == "BYTES":
        array = triton_utils.deserialize_bytes_tensor(raw)
      else:
        array = np.frombuffer(raw, dtype=dtype)
    else:
      field = _CONTENTS_FIELDS.get(tensor.datatype)
      if field is None:
        raise _RequestError(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"datatype '{tensor.datatype}' of input '{tensor.name}' requires"
            " raw_input_contents",
        )
      values = list(getattr(tensor.contents, field))
      array = np.array(values, dtype=dtype)
    return array.reshape(shape)
  except ValueError as e:
    raise _RequestError(
        grpc.StatusCode.INVALID_ARGUMENT,
        f"unexpected size of input '{tensor.name}' for shape {shape}: {e}",
    ) from e


def encode_tensor(array: np.ndarray) -> tuple[str, bytes]:
  """Returns the datatype and raw bytes of an output tensor."""
  array = np.asarray(array)
  if array.dtype.kind in "OSU":
    return "BYTES", triton_utils.serialize_byte_tensor(array).item()
  datatype = triton_utils.np_to_triton_dtype(array.dtype)
  if datatype is None:
    raise _RequestError(
        grpc.StatusCode.INTERNAL, f"unsupported output dtype {array.dtype}"
    )
  return datatype, np.ascontiguousarray(array).tobytes()


class InferenceServicer(service_pb2_grpc.GRPCInferenceServiceServicer):
  """Serves the models of a ModelRepository."""

  def __init__(
      self,
      repository: models.ModelRepository,
      shm_manager: shared_memory.SharedMemoryManager | None = None,
      allow_shared_memory: bool = True,
  ) -> None:
    """Initializes the servicer.

    Args:
      repository: The models to serve.
      shm_manager: Manager of system shared memory regions.
      allow_shared_memory: Whether clients may register shared memory
        regions. If not, clients send tensors inline.
    """
    self._repository = repository
    self._shm = shm_manager or shared_memory.SharedMemoryManager()
    self._allow_shared_memory = allow_shared_memory
    self._ready = True

  def set_ready(self, ready: bool) -> None:
    self._ready = ready

  # Health and metadata.

  def ServerLive(self, request, context):
    return service_pb2.ServerLiveResponse(live=True)

  def ServerReady(self, request, context):
    return service_pb2.ServerReadyResponse(ready=self._ready)

  def ServerMetadata(self, request, context):
    return service_pb2.ServerMetadataResponse(
        name=SERVER_NAME, version=SERVER_VERSION, extensions=_EXTENSIONS
    )

  def ModelReady(self, request, context):
    return service_pb2.ModelReadyResponse(
        ready=self._repository.is_ready(request.name, request.version)
    )

  @_handle_errors
  def ModelMetadata(self, request, context):
    model = self._repository.get(request.name, request.version)
    tensor_metadata = service_pb2.ModelMetadataResponse.TensorMetadata
    return service_pb2.ModelMetadataResponse(
        name=model.name,
        versions=[model.version],
        platform=model.platform,
        inputs=[
            tensor_metadata(
                name=s.name,
                datatype=triton_utils.np_to_triton_dtype(s.dtype),
                shape=s.shape,
            )
            for s in model.inputs
        ],
        outputs=[
            tensor_metadata(
                name=s.name,
                datatype=triton_utils.np_to_triton_dtype(s.dtype),
                shape=s.shape,
            )
            for s in model.outputs
        ],
    )

  @_handle_errors
  def ModelConfig(self, request, context):
    model = self._repository.get(request.name, request.version)
    return service_pb2.ModelConfigResponse(config=model.config)

  # Inference.

  @_handle_errors
  def ModelInfer(self, request, context):
    model = self._repository.get(request.model_name, request.model_version)
    inputs = self._decode_inputs(request)
    requested = [o.name for o in request.outputs]
    outputs = model.infer(inputs, requested)

    response = service_pb2.ModelInferResponse(
        model_name=model.name, model_version=model.version, id=request.id
    )
    requested_by_name = {o.name: o for o in request.outputs}
    for name, array in outputs.items():
      datatype, data = encode_tensor(array)
      output = response.outputs.add(
          name=name, datatype=datatype, shape=list(np.shape(array))
      )
      requested_output = requested_by_name.get(name)
      shm = (
          _shm_params(requested_output.parameters)
          if requested_output is not None
          else None
      )
      if shm is not None:
        region, offset, byte_size = shm
        self._shm.write(region, offset, data, byte_size)
        output.parameters[_SHM_REGION].string_param = region
        output.parameters[_SHM_OFFSET].int64_param = offset
        output.parameters[_SHM_BYTE_SIZE].int64_param = len(data)
      else:
        response.raw_output_contents.append(data)
    return response

  def _decode_inputs(
      self, request: service_pb2.ModelInferRequest
  ) -> dict[str, np.ndarray]:
    """Returns the request's inputs from raw, typed or shared memory data."""
    inline_inputs = [
        t for t in request.inputs if _SHM_REGION not in t.parameters
    ]
    raw_contents = list(request.raw_input_contents)
    if raw_contents and len(raw_contents) != len(inline_inputs):
      raise _RequestError(
          grpc.StatusCode.INVALID_ARGUMENT,
          f"expected {len(inline_inputs)} raw_input_contents, got"
          f" {len(raw_contents)}",
      )
    raw_iter = iter(raw_contents)
    inputs = {}
    for tensor in request.inputs:
      shm = _shm_params(tensor.parameters)
      if shm is not None:
        raw = self._shm.read(*shm)
      elif raw_contents:
        raw = next(raw_iter)
      else:
        raw = None
      inputs[tensor.name] = decode_tensor(tensor, raw)
    return inputs

  # Explicit model control.

  @_handle_errors
  def RepositoryIndex(self, request, context):
    response = service_pb2.RepositoryIndexResponse()
    for status in self._repository.index():
      if request.ready and status.state != "READY":
        continue
      response.models.add(
          name=status.name,
          version=status.version,
          state=status.state,
          reason=status.reason,
      )
    return response

  @_handle_errors
  def RepositoryModelLoad(self, request, context):
    config_json = None
    if "config" in request.parameters:
      config_json = request.parameters["config"].string_param
    self._repository.load(request.model_name, config_json)
    return service_pb2.RepositoryModelLoadResponse()

  @_handle_errors
  def RepositoryModelUnload(self, request, context):
    self._repository.unload(request.model_name)
    return service_pb2.RepositoryModelUnloadResponse()

  # System shared memory.

  @_handle_errors
  def SystemSharedMemoryRegister(self, request, context):
    if not self._allow_shared_memory:
      raise _RequestError(
          grpc.StatusCode.UNAVAILABLE,
          "system shared memory is disabled (--allow-client-shm=false)",
      )
    self._shm.register(
        request.name, request.key, request.offset, request.byte_size
    )
    return service_pb2.SystemSharedMemoryRegisterResponse()

  @_handle_errors
  def SystemSharedMemoryStatus(self, request, context):
    response = service_pb2.SystemSharedMemoryStatusResponse()
    for region in self._shm.status(request.name):
      response.regions[region.name].CopyFrom(
          service_pb2.SystemSharedMemoryStatusResponse.RegionStatus(
              name=region.name,
              key=region.key,
              offset=region.offset,
              byte_size=region.byte_size,
          )
      )
    return response

  @_handle_errors
  def SystemSharedMemoryUnregister(self, request, context):
    self._shm.unregister(request.name)
    return service_pb2.SystemSharedMemoryUnregisterResponse()
