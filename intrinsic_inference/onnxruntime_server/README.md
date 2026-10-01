# ONNX Runtime inference server

The inference service runs its models with this server. It replaces Triton
Inference Server, which is only available for x86-64 and needs an NVIDIA GPU.
This server runs on x86-64 and arm64, with or without an NVIDIA GPU.

It speaks the same [Open Inference Protocol](https://github.com/kserve/open-inference-protocol)
gRPC API as Triton and loads the same model repositories, so the inference
service and its clients work unchanged.

## Devices

`--device` selects where models run:

* `auto` (default): uses the CUDA execution provider if an NVIDIA driver and
  GPU are available, and the CPU otherwise.
* `cpu`: always uses the CPU.
* `gpu`: fails to load models if no GPU execution provider can be used.

Python backend models see the selected device in the
`INTRINSIC_INFERENCE_DEVICE` environment variable.

In the cluster, the server gets the GPU without requesting `nvidia.com/gpu`
from Kubernetes. On hosts set up with
[`setup_nvidia.sh`](../../intrinsic_runtime/setup_nvidia.sh), NVIDIA's
container runtime is the default, and the image's
`NVIDIA_VISIBLE_DEVICES=all` exposes the GPUs. On other hosts, the same
deployment runs on the CPU.

## Model repositories

The server loads [Triton model repositories](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
with these backends:

* `onnxruntime`: ONNX models. `config.pbtxt` is optional; missing inputs and
  outputs are taken from the model.
* `python`: a `model.py` defining `TritonPythonModel`, as for Triton's Python
  backend. The server provides `triton_python_backend_utils`. Triton's
  `EXECUTION_ENV_PATH` environments aren't supported, so models can only
  import packages in the server image (numpy, OpenCV, trimesh, onnxruntime)
  and modules next to `model.py`.

## API

Supported: health (`ServerLive`, `ServerReady`), metadata (`ServerMetadata`,
`ModelMetadata`, `ModelConfig`, `ModelReady`), `ModelInfer`, the model
repository API (`RepositoryIndex`, `RepositoryModelLoad`,
`RepositoryModelUnload`) and system shared memory
(`SystemSharedMemoryRegister`, `SystemSharedMemoryStatus`,
`SystemSharedMemoryUnregister`). Other RPCs return `UNIMPLEMENTED`.

The server accepts `tritonserver`'s flags. Flags that only apply to Triton,
such as `--trace-config` and `--metrics-port`, are ignored.

## Run locally

```bash
bazel run //intrinsic_inference/onnxruntime_server:server_main -- \
  --model-repository=/path/to/models --grpc-address=0.0.0.0 --grpc-port=8001
```

## Tests

```bash
bazel test //intrinsic_inference/onnxruntime_server/...
```

To also run the released RF-DETR segmentation model, point
`SEGMENTATION_MODEL_DIR` to a directory containing its `config.pbtxt` and
`segmentation.onnx`:

```bash
bazel test //intrinsic_inference/onnxruntime_server:server_test \
  --test_env=SEGMENTATION_MODEL_DIR=/path/to/segmentation
```
