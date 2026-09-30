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

"""Tests the server over gRPC, using the same client code as the inference service."""

import os
import shutil
import tempfile
import time

from absl import logging
from absl.testing import absltest
import cv2
import grpc
import numpy as np
from tritonclient.grpc import model_config_pb2
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
from tritonclient import utils as triton_utils

from intrinsic_inference.core import triton_shm_utils
from intrinsic_inference.onnxruntime_server import server_main
from intrinsic_inference.onnxruntime_server import testing_models


def _infer_request(model_name, inputs, outputs=()):
  """Builds a ModelInferRequest with raw inputs, like the inference service."""
  request = service_pb2.ModelInferRequest(model_name=model_name)
  for name, array in inputs.items():
    request.inputs.add(
        name=name,
        datatype=triton_utils.np_to_triton_dtype(array.dtype),
        shape=array.shape,
    )
    if array.dtype == np.object_:
      request.raw_input_contents.append(
          triton_utils.serialize_byte_tensor(array).item()
      )
    else:
      request.raw_input_contents.append(array.tobytes())
  for name in outputs:
    request.outputs.add(name=name)
  return request


def _outputs(response):
  """Returns the response's raw outputs as numpy arrays by name."""
  result = {}
  for output, raw in zip(response.outputs, response.raw_output_contents):
    if output.datatype == "BYTES":
      array = triton_utils.deserialize_bytes_tensor(raw)
    else:
      array = np.frombuffer(
          raw, dtype=triton_utils.triton_to_np_dtype(output.datatype)
      )
    result[output.name] = array.reshape(output.shape)
  return result


class ServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.repo_path = self.create_tempdir("models").full_path
    testing_models.write_segmentation_like_model(self.repo_path, "seg")
    testing_models.write_python_model(self.repo_path, "py")
    testing_models.write_static_add_model(self.repo_path, "add")
    # Unix socket paths are limited to 108 characters, so don't use the test's
    # temporary directory, whose path contains the test name.
    socket_dir = tempfile.mkdtemp(prefix="ort")
    self.addCleanup(shutil.rmtree, socket_dir)
    socket = os.path.join(socket_dir, "triton.sock")
    # The flags with which the inference service starts Triton.
    args = server_main.parse_args([
        f"--model-repository={self.repo_path}",
        "--allow-client-shm=true",
        "--allow-http=false",
        "--allow-grpc=true",
        f"--grpc-address=unix://{socket}",
        "--grpc-port=0",
        "--metrics-port=8002",
        "--model-control-mode=explicit",
        "--trace-config=mode=opentelemetry",
        "--device=cpu",
    ])
    self.server, self.servicer, self.repository = server_main.build_server(
        args
    )
    self.server.start()
    self.addCleanup(self.server.stop, None)
    self.channel = grpc.insecure_channel(f"unix://{socket}")
    self.addCleanup(self.channel.close)
    self.stub = service_pb2_grpc.GRPCInferenceServiceStub(self.channel)

  def _load(self, name):
    self.stub.RepositoryModelLoad(
        service_pb2.RepositoryModelLoadRequest(model_name=name)
    )

  def test_health_and_metadata(self):
    self.assertTrue(
        self.stub.ServerLive(service_pb2.ServerLiveRequest()).live
    )
    self.assertTrue(
        self.stub.ServerReady(service_pb2.ServerReadyRequest()).ready
    )
    metadata = self.stub.ServerMetadata(service_pb2.ServerMetadataRequest())
    self.assertIn("system_shared_memory", metadata.extensions)

  def test_explicit_mode_loads_models_on_request(self):
    ready = lambda: self.stub.ModelReady(
        service_pb2.ModelReadyRequest(name="seg")
    ).ready
    self.assertFalse(ready())
    self._load("seg")
    self.assertTrue(ready())

    index = self.stub.RepositoryIndex(service_pb2.RepositoryIndexRequest())
    states = {m.name: m.state for m in index.models}
    self.assertEqual(
        states, {"seg": "READY", "py": "UNAVAILABLE", "add": "UNAVAILABLE"}
    )
    ready_only = self.stub.RepositoryIndex(
        service_pb2.RepositoryIndexRequest(ready=True)
    )
    self.assertEqual([m.name for m in ready_only.models], ["seg"])

    self.stub.RepositoryModelUnload(
        service_pb2.RepositoryModelUnloadRequest(model_name="seg")
    )
    self.assertFalse(ready())

  def test_load_failure_is_reported(self):
    with self.assertRaises(grpc.RpcError) as e:
      self._load("does_not_exist")
    self.assertEqual(e.exception.code(), grpc.StatusCode.NOT_FOUND)
    self.assertIn("failed to load 'does_not_exist'", e.exception.details())

  def test_load_with_config_parameter(self):
    request = service_pb2.RepositoryModelLoadRequest(model_name="seg")
    request.parameters["config"].string_param = (
        '{"name": "seg", "backend": "onnxruntime"}'
    )
    self.stub.RepositoryModelLoad(request)
    config = self.stub.ModelConfig(
        service_pb2.ModelConfigRequest(name="seg")
    ).config
    self.assertEmpty(config.instance_group)
    self.assertLen(config.input, 2)

  def test_model_metadata_and_config(self):
    self._load("seg")
    metadata = self.stub.ModelMetadata(
        service_pb2.ModelMetadataRequest(name="seg")
    )
    self.assertEqual(metadata.platform, "onnxruntime_onnx")
    self.assertEqual(list(metadata.versions), ["1"])
    self.assertEqual(
        [(t.name, t.datatype, list(t.shape)) for t in metadata.outputs],
        [("mean", "FP32", [1, 3, -1, -1]), ("mask", "BOOL", [1, 3, -1, -1])],
    )
    config = self.stub.ModelConfig(
        service_pb2.ModelConfigRequest(name="seg")
    ).config
    self.assertEqual(config.input[0].data_type, model_config_pb2.TYPE_UINT8)

  def test_infer_onnx_model(self):
    self._load("seg")
    image = np.random.default_rng(0).integers(
        0, 255, (1, 3, 6, 8), dtype=np.uint8
    )
    thresholds = np.array([[0.5, 60.0]], np.float32)
    response = self.stub.ModelInfer(
        _infer_request(
            "seg", {"input": image, "thresholds": thresholds}, ["mean", "mask"]
        )
    )
    self.assertEqual(response.model_version, "1")
    outputs = _outputs(response)
    np.testing.assert_allclose(outputs["mean"], image * 0.5)
    np.testing.assert_array_equal(outputs["mask"], image * 0.5 > 60.0)

  def test_infer_python_model_with_bytes_output(self):
    self._load("py")
    response = self.stub.ModelInfer(
        _infer_request("py", {"VALUES": np.array([1, 2, 3], np.float32)})
    )
    outputs = _outputs(response)
    np.testing.assert_array_equal(outputs["SUM"], [6.0])
    self.assertEqual(outputs["DEVICE"].tolist(), [b"cpu"])

  def test_infer_with_typed_contents(self):
    self._load("add")
    request = service_pb2.ModelInferRequest(model_name="add")
    for name, values in [("X", range(6)), ("Y", [10] * 6)]:
      tensor = request.inputs.add(name=name, datatype="FP32", shape=[2, 3])
      tensor.contents.fp32_contents.extend(values)
    outputs = _outputs(self.stub.ModelInfer(request))
    np.testing.assert_array_equal(
        outputs["SUM"], np.arange(6).reshape(2, 3) + 10
    )

  def test_infer_errors(self):
    self._load("seg")
    with self.assertRaises(grpc.RpcError) as e:
      self.stub.ModelInfer(
          _infer_request("unknown", {"x": np.zeros(1, np.float32)})
      )
    self.assertEqual(e.exception.code(), grpc.StatusCode.NOT_FOUND)

    with self.assertRaises(grpc.RpcError) as e:
      self.stub.ModelInfer(
          _infer_request(
              "seg",
              {
                  "input": np.zeros((1, 3, 2, 2), np.float32),
                  "thresholds": np.zeros((1, 2), np.float32),
              },
          )
      )
    self.assertEqual(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    request = _infer_request(
        "seg",
        {
            "input": np.zeros((1, 3, 2, 2), np.uint8),
            "thresholds": np.zeros((1, 2), np.float32),
        },
    )
    request.raw_input_contents[0] = b"\x00"
    with self.assertRaises(grpc.RpcError) as e:
      self.stub.ModelInfer(request)
    self.assertEqual(e.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

  def test_python_model_error_is_internal(self):
    self._load("py")
    with self.assertRaises(grpc.RpcError) as e:
      self.stub.ModelInfer(
          _infer_request("py", {"VALUES": np.array([-1], np.float32)})
      )
    self.assertEqual(e.exception.code(), grpc.StatusCode.INTERNAL)
    self.assertIn("negative values", e.exception.details())

  def test_shared_memory_inference_like_inference_service(self):
    self._load("seg")
    self._load("add")
    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=self.stub, pool_size=2, byte_size=1 << 20
    )
    self.addCleanup(pool.cleanup)
    status = self.stub.SystemSharedMemoryStatus(
        service_pb2.SystemSharedMemoryStatusRequest()
    )
    self.assertLen(status.regions, 4)

    # Dynamic output shapes: inputs via shared memory, outputs inline.
    image = np.random.default_rng(1).integers(
        0, 255, (1, 3, 5, 7), dtype=np.uint8
    )
    request = _infer_request(
        "seg",
        {"input": image, "thresholds": np.array([[2.0, 100.0]], np.float32)},
    )
    config = self.repository.get("seg").config
    response = triton_shm_utils.run_inference(
        request=request, pool=pool, stub=self.stub, model_config=config
    )
    self.assertEmpty(request.raw_input_contents)  # Sent via shared memory.
    outputs = _outputs(response)
    np.testing.assert_allclose(outputs["mean"], image * 2.0)

    # Static output shapes: inputs and outputs via shared memory.
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    request = _infer_request("add", {"X": x, "Y": x})
    config = self.repository.get("add").config
    response = triton_shm_utils.run_inference(
        request=request, pool=pool, stub=self.stub, model_config=config
    )
    self.assertIn("shared_memory_region", request.outputs[0].parameters)
    np.testing.assert_array_equal(_outputs(response)["SUM"], 2 * x)

  def test_invalid_shared_memory_access_is_shared_memory_error(self):
    self._load("add")
    request = _infer_request(
        "add",
        {"X": np.zeros((2, 3), np.float32), "Y": np.zeros((2, 3), np.float32)},
    )
    request.ClearField("raw_input_contents")
    for tensor in request.inputs:
      tensor.parameters["shared_memory_region"].string_param = "missing"
      tensor.parameters["shared_memory_byte_size"].int64_param = 24
    with self.assertRaises(grpc.RpcError) as e:
      self.stub.ModelInfer(request)
    # The inference service falls back to inline tensors on such errors.
    self.assertTrue(triton_shm_utils.is_shm_error(e.exception))

  def test_register_shared_memory_twice_fails(self):
    pool = triton_shm_utils.RawGrpcSharedMemoryPool(
        stub=self.stub, pool_size=1, byte_size=1024
    )
    self.addCleanup(pool.cleanup)
    with self.assertRaises(grpc.RpcError):
      self.stub.SystemSharedMemoryRegister(
          service_pb2.SystemSharedMemoryRegisterRequest(
              name="in_region_0", key="/in_shm_0", byte_size=1024
          )
      )


class ServerStartupTest(absltest.TestCase):

  def test_none_mode_loads_all_models(self):
    repo_path = self.create_tempdir().full_path
    testing_models.write_segmentation_like_model(repo_path, "seg")
    testing_models.write_python_model(repo_path, "py")
    args = server_main.parse_args([
        f"--model-repository={repo_path}",
        "--grpc-address=127.0.0.1",
        "--grpc-port=0",
        "--device=cpu",
    ])
    _, _, repository = server_main.build_server(args)
    self.assertTrue(repository.is_ready("seg"))
    self.assertTrue(repository.is_ready("py"))

  def test_explicit_mode_with_load_model(self):
    repo_path = self.create_tempdir().full_path
    testing_models.write_segmentation_like_model(repo_path, "seg")
    testing_models.write_python_model(repo_path, "py")
    args = server_main.parse_args([
        f"--model-repository={repo_path}",
        "--grpc-address=127.0.0.1",
        "--grpc-port=0",
        "--model-control-mode=explicit",
        "--load-model=py",
    ])
    _, _, repository = server_main.build_server(args)
    self.assertFalse(repository.is_ready("seg"))
    self.assertTrue(repository.is_ready("py"))

  def test_grpc_endpoint(self):
    self.assertEqual(
        server_main.grpc_endpoint("unix:///dev/shm/triton.sock", 0),
        "unix:///dev/shm/triton.sock",
    )
    self.assertEqual(
        server_main.grpc_endpoint("0.0.0.0", 8001), "0.0.0.0:8001"
    )


_SEGMENTATION_MODEL_DIR_ENV = "SEGMENTATION_MODEL_DIR"
_SEGMENTATION_MODEL = "ai.intrinsic.ioc_pose_estimation.segmentor.rfdetr"


@absltest.skipUnless(
    os.environ.get(_SEGMENTATION_MODEL_DIR_ENV),
    f"{_SEGMENTATION_MODEL_DIR_ENV} is not set",
)
class RealSegmentationModelTest(absltest.TestCase):
  """Serves the real RF-DETR segmentation model and segments a drawn block.

  SEGMENTATION_MODEL_DIR must contain the released config.pbtxt and
  segmentation.onnx.
  """

  def test_segments_block(self):
    source = os.environ[_SEGMENTATION_MODEL_DIR_ENV]
    model_dir = os.path.join(
        self.create_tempdir("models").full_path, _SEGMENTATION_MODEL
    )
    os.makedirs(os.path.join(model_dir, "1"))
    os.symlink(
        os.path.join(source, "config.pbtxt"),
        os.path.join(model_dir, "config.pbtxt"),
    )
    os.symlink(
        os.path.join(source, "segmentation.onnx"),
        os.path.join(model_dir, "1", "segmentation.onnx"),
    )
    socket_dir = tempfile.mkdtemp(prefix="ort")
    self.addCleanup(shutil.rmtree, socket_dir)
    socket = os.path.join(socket_dir, "triton.sock")
    server, _, _ = server_main.build_server(
        server_main.parse_args([
            f"--model-repository={os.path.dirname(model_dir)}",
            f"--grpc-address=unix://{socket}",
            "--model-control-mode=none",
            "--device=auto",
        ])
    )
    server.start()
    self.addCleanup(server.stop, None)
    channel = grpc.insecure_channel(f"unix://{socket}")
    self.addCleanup(channel.close)
    stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    self.assertTrue(
        stub.ModelReady(
            service_pb2.ModelReadyRequest(name=_SEGMENTATION_MODEL)
        ).ready
    )

    # A shaded block (top, front and side faces) on a gray table.
    image = np.full((480, 640, 3), 90, np.uint8)
    faces = [
        (np.array([[260, 180], [380, 160], [430, 210], [310, 232]]), 205),
        (np.array([[310, 232], [430, 210], [430, 300], [310, 325]]), 160),
        (np.array([[260, 180], [310, 232], [310, 325], [260, 270]]), 120),
    ]
    expected_mask = np.zeros((480, 640), np.uint8)
    for polygon, gray in faces:
      cv2.fillPoly(image, [polygon], (gray, gray + 3, gray + 6))
      cv2.fillPoly(expected_mask, [polygon], 1)

    # The same request as SegmentationModel.run_inference.
    request = _infer_request(
        _SEGMENTATION_MODEL,
        {
            "input": np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
            "thresholds": np.array([[0.5, 0.5]], np.float32),
        },
        outputs=("boxes", "scores", "masks", "visibility"),
    )
    start = time.perf_counter()
    outputs = _outputs(stub.ModelInfer(request))
    logging.info("Segmentation took %.2f s", time.perf_counter() - start)

    self.assertEqual(outputs["boxes"].shape, (1, 4))
    self.assertEqual(outputs["masks"].shape, (1, 480, 640))
    self.assertEqual(outputs["masks"].dtype, np.bool_)
    self.assertGreater(outputs["scores"][0], 0.9)
    self.assertGreater(outputs["visibility"][0], 0.5)
    np.testing.assert_allclose(
        outputs["boxes"][0], [260, 160, 430, 325], atol=5
    )
    mask = outputs["masks"][0]
    iou = (mask & (expected_mask > 0)).sum() / (mask | (expected_mask > 0)).sum()
    self.assertGreater(iou, 0.9)


if __name__ == "__main__":
  absltest.main()
