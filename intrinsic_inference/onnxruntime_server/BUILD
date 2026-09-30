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

# An ONNX Runtime based replacement for Triton Inference Server that runs on
# CPUs and GPUs of any architecture. It serves Triton model repositories over
# the same Open Inference Protocol gRPC API.

load("//bazel:python.bzl", "py_binary", "py_library", "py_test")
load("//bazel:python_oci_image.bzl", "python_oci_image")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "execution_providers",
    srcs = ["execution_providers.py"],
    deps = [
        "@ai_intrinsic_sdks_pip_deps//onnxruntime_gpu",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "triton_python_backend_utils",
    srcs = ["triton_python_backend_utils.py"],
    deps = ["@ai_intrinsic_sdks_pip_deps//numpy"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [
        ":execution_providers",
        ":triton_python_backend_utils",
        "@ai_intrinsic_sdks_pip_deps//numpy",
        "@ai_intrinsic_sdks_pip_deps//tritonclient",
        "@com_google_absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "shared_memory",
    srcs = ["shared_memory.py"],
)

py_library(
    name = "server",
    srcs = ["server.py"],
    deps = [
        ":models",
        ":shared_memory",
        "@ai_intrinsic_sdks_pip_deps//numpy",
        "@ai_intrinsic_sdks_pip_deps//tritonclient",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "server_main_lib",
    srcs = ["server_main.py"],
    deps = [
        ":execution_providers",
        ":models",
        ":server",
        "@ai_intrinsic_sdks_pip_deps//tritonclient",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
    ],
)

py_binary(
    name = "server_main",
    srcs = ["server_main.py"],
    deps = [
        ":server_main_lib",
        # CUDA and cuDNN for the CUDA execution provider.
        "@ai_intrinsic_sdks_pip_deps//nvidia_cublas",
        "@ai_intrinsic_sdks_pip_deps//nvidia_cuda_nvrtc",
        "@ai_intrinsic_sdks_pip_deps//nvidia_cuda_runtime",
        "@ai_intrinsic_sdks_pip_deps//nvidia_cudnn_cu13",
        "@ai_intrinsic_sdks_pip_deps//nvidia_cufft",
        "@ai_intrinsic_sdks_pip_deps//nvidia_curand",
        "@ai_intrinsic_sdks_pip_deps//nvidia_nvjitlink",
        # Used by Python backend models (e.g. FoundationPose). Triton's
        # EXECUTION_ENV_PATH environments are not supported.
        "@ai_intrinsic_sdks_pip_deps//opencv_python_headless",
        "@ai_intrinsic_sdks_pip_deps//trimesh",
    ],
)

python_oci_image(
    name = "onnxruntime_server_image",
    base = "@distroless_python3",
    binary = "server_main",
)

py_library(
    name = "testing_models",
    testonly = True,
    srcs = ["testing_models.py"],
    deps = [
        "@ai_intrinsic_sdks_pip_deps//numpy",
        "@ai_intrinsic_sdks_pip_deps//onnx",
    ],
)

py_test(
    name = "execution_providers_test",
    srcs = ["execution_providers_test.py"],
    deps = [
        ":execution_providers",
        ":testing_models",
        "@ai_intrinsic_sdks_pip_deps//onnxruntime_gpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":execution_providers",
        ":models",
        ":testing_models",
        "@ai_intrinsic_sdks_pip_deps//numpy",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    # The real segmentation model runs if SEGMENTATION_MODEL_DIR is passed with
    # --test_env.
    deps = [
        ":server_main_lib",
        ":testing_models",
        "//intrinsic_inference/core:triton_shm_utils",
        "@ai_intrinsic_sdks_pip_deps//numpy",
        "@ai_intrinsic_sdks_pip_deps//opencv_python_headless",
        "@ai_intrinsic_sdks_pip_deps//tritonclient",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
