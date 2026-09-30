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

"""Writes small models into a Triton model repository for tests."""

import os
import textwrap

import onnx
from onnx import helper
from onnx import TensorProto

# Mirrors the RF-DETR segmentation model's interface: a uint8 image with
# dynamic height and width plus thresholds, producing float and bool outputs.
SEGMENTATION_LIKE_CONFIG = """
name: "{name}"
backend: "onnxruntime"
default_model_filename: "model.onnx"
input [
  {{ name: "input" data_type: TYPE_UINT8 dims: [1, 3, -1, -1] }},
  {{ name: "thresholds" data_type: TYPE_FP32 dims: [1, 2] }}
]
output [
  {{ name: "mean" data_type: TYPE_FP32 dims: [1, 3, -1, -1] }},
  {{ name: "mask" data_type: TYPE_BOOL dims: [1, 3, -1, -1] }}
]
instance_group [ {{ count: 1 kind: KIND_GPU }} ]
"""


def _segmentation_like_onnx() -> onnx.ModelProto:
  """Returns a model computing mean = input * thresholds[0, 0] and mask = mean > thresholds[0, 1]."""
  nodes = [
      helper.make_node("Cast", ["input"], ["input_f"], to=TensorProto.FLOAT),
      helper.make_node(
          "Slice", ["thresholds", "s0", "s1", "axis1"], ["scale"]
      ),
      helper.make_node(
          "Slice", ["thresholds", "s1", "s2", "axis1"], ["threshold"]
      ),
      helper.make_node("Reshape", ["scale", "shape1"], ["scale_r"]),
      helper.make_node("Reshape", ["threshold", "shape1"], ["threshold_r"]),
      helper.make_node("Mul", ["input_f", "scale_r"], ["mean"]),
      helper.make_node("Greater", ["mean", "threshold_r"], ["mask"]),
  ]
  initializers = [
      helper.make_tensor("s0", TensorProto.INT64, [1], [0]),
      helper.make_tensor("s1", TensorProto.INT64, [1], [1]),
      helper.make_tensor("s2", TensorProto.INT64, [1], [2]),
      helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
      helper.make_tensor("shape1", TensorProto.INT64, [1], [1]),
  ]
  graph = helper.make_graph(
      nodes,
      "segmentation_like",
      [
          helper.make_tensor_value_info(
              "input", TensorProto.UINT8, [1, 3, "height", "width"]
          ),
          helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, [1, 2]),
      ],
      [
          helper.make_tensor_value_info(
              "mean", TensorProto.FLOAT, [1, 3, "height", "width"]
          ),
          helper.make_tensor_value_info(
              "mask", TensorProto.BOOL, [1, 3, "height", "width"]
          ),
      ],
      initializer=initializers,
  )
  return helper.make_model(
      graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
  )


def write_segmentation_like_model(
    repository: str, name: str, version: str = "1", with_config: bool = True
) -> str:
  """Writes the segmentation-like ONNX model and returns its directory."""
  model_dir = os.path.join(repository, name)
  os.makedirs(os.path.join(model_dir, version), exist_ok=True)
  onnx.save(
      _segmentation_like_onnx(), os.path.join(model_dir, version, "model.onnx")
  )
  if with_config:
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
      f.write(SEGMENTATION_LIKE_CONFIG.format(name=name))
  return model_dir


# A Python backend model using the triton_python_backend_utils API like
# FoundationPose's model.py: SUM = sum(VALUES) * SCALE, with SCALE optional.
PYTHON_MODEL_PY = '''
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

  def initialize(self, args):
    self.args = args
    self.config = json.loads(args["model_config"])
    self.device = os.environ.get("INTRINSIC_INFERENCE_DEVICE")
    marker = os.path.join(args["model_repository"], "initialized")
    open(marker, "w").close()

  def execute(self, requests):
    responses = []
    for request in requests:
      values = pb_utils.get_input_tensor_by_name(request, "VALUES").as_numpy()
      scale_tensor = pb_utils.get_input_tensor_by_name(request, "SCALE")
      scale = 1 if scale_tensor is None else scale_tensor.as_numpy()[0]
      if np.any(values < 0):
        responses.append(pb_utils.InferenceResponse(
            output_tensors=[],
            error=pb_utils.TritonError("negative values are not supported")))
        continue
      responses.append(pb_utils.InferenceResponse(output_tensors=[
          pb_utils.Tensor("SUM", np.array([values.sum() * scale], np.float32)),
          pb_utils.Tensor("DEVICE", np.array([self.device.encode()], np.object_)),
      ]))
    return responses

  def finalize(self):
    marker = os.path.join(self.args["model_repository"], "finalized")
    open(marker, "w").close()
'''

PYTHON_MODEL_CONFIG = """
name: "{name}"
backend: "python"
input [
  {{ name: "VALUES" data_type: TYPE_FP32 dims: [-1] }},
  {{ name: "SCALE" data_type: TYPE_INT32 dims: [1] optional: true }}
]
output [
  {{ name: "SUM" data_type: TYPE_FP32 dims: [1] }},
  {{ name: "DEVICE" data_type: TYPE_STRING dims: [1] }}
]
parameters {{
  key: "EXECUTION_ENV_PATH"
  value {{ string_value: "$$TRITON_MODEL_DIRECTORY/env.tar.gz" }}
}}
"""


def write_python_model(repository: str, name: str, version: str = "1") -> str:
  """Writes the Python backend model and returns its directory."""
  model_dir = os.path.join(repository, name)
  os.makedirs(os.path.join(model_dir, version), exist_ok=True)
  with open(os.path.join(model_dir, version, "model.py"), "w") as f:
    f.write(textwrap.dedent(PYTHON_MODEL_PY))
  with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
    f.write(PYTHON_MODEL_CONFIG.format(name=name))
  return model_dir


STATIC_ADD_CONFIG = """
name: "{name}"
platform: "onnxruntime_onnx"
input [
  {{ name: "X" data_type: TYPE_FP32 dims: [2, 3] }},
  {{ name: "Y" data_type: TYPE_FP32 dims: [2, 3] }}
]
output [
  {{ name: "SUM" data_type: TYPE_FP32 dims: [2, 3] }}
]
"""


def write_static_add_model(repository: str, name: str) -> str:
  """Writes a model with static shapes computing SUM = X + Y."""
  graph = helper.make_graph(
      [helper.make_node("Add", ["X", "Y"], ["SUM"])],
      "static_add",
      [
          helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3]),
          helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3]),
      ],
      [helper.make_tensor_value_info("SUM", TensorProto.FLOAT, [2, 3])],
  )
  model = helper.make_model(
      graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
  )
  model_dir = os.path.join(repository, name)
  os.makedirs(os.path.join(model_dir, "1"), exist_ok=True)
  onnx.save(model, os.path.join(model_dir, "1", "model.onnx"))
  with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
    f.write(STATIC_ADD_CONFIG.format(name=name))
  return model_dir
