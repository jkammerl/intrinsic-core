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

"""Runs the ONNX Runtime inference server.

Accepts the command line flags of `tritonserver` that the inference service
uses, so that it can be started with the same arguments as Triton, e.g.:

  server_main --model-repository=/models \
      --grpc-address=unix:///dev/shm/triton.sock --grpc-port=0 \
      --model-control-mode=explicit --device=auto

Flags that only apply to Triton (e.g. --trace-config, --metrics-port) are
accepted and ignored.
"""

import argparse
from collections.abc import Sequence
from concurrent import futures
import signal
import sys
import threading

from absl import logging
import grpc
from tritonclient.grpc import service_pb2_grpc

from intrinsic_inference.onnxruntime_server import execution_providers
from intrinsic_inference.onnxruntime_server import models
from intrinsic_inference.onnxruntime_server import server

_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", -1),
    ("grpc.max_send_message_length", -1),
]


def _bool_flag(value: str) -> bool:
  if value.lower() in ("1", "true", "yes", "on"):
    return True
  if value.lower() in ("0", "false", "no", "off"):
    return False
  raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Parses the command line, ignoring unknown (Triton-only) flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-repository", required=True)
  parser.add_argument("--grpc-address", default="0.0.0.0")
  parser.add_argument("--grpc-port", type=int, default=8001)
  parser.add_argument(
      "--model-control-mode",
      choices=["none", "explicit", "poll"],
      default="none",
      help=(
          "none: load all models at startup; explicit: load models only on"
          " request or with --load-model."
      ),
  )
  parser.add_argument(
      "--load-model",
      action="append",
      default=[],
      help="Model to load at startup in explicit mode, '*' for all.",
  )
  parser.add_argument(
      "--device",
      choices=[d.value for d in execution_providers.Device],
      default=execution_providers.Device.AUTO.value,
      help="Where to run inference; auto uses a GPU if one is available.",
  )
  parser.add_argument("--allow-grpc", type=_bool_flag, default=True)
  parser.add_argument("--allow-client-shm", type=_bool_flag, default=True)
  parser.add_argument("--max-workers", type=int, default=8)
  args, unknown = parser.parse_known_args(argv)
  if unknown:
    logging.info("Ignoring unsupported tritonserver flags: %s", unknown)
  return args


def grpc_endpoint(address: str, port: int) -> str:
  """Returns the gRPC endpoint for Triton's --grpc-address/--grpc-port.

  Like Triton, appends the port even to unix socket addresses: with
  --grpc-address=unix:///dev/shm/triton.sock --grpc-port=0 the socket file is
  /dev/shm/triton.sock:0, which is where the inference service connects.
  """
  return f"{address}:{port}"


def build_server(
    args: argparse.Namespace,
) -> tuple[grpc.Server, server.InferenceServicer, models.ModelRepository]:
  """Creates the (not yet started) gRPC server and loads startup models."""
  device = execution_providers.Device(args.device)
  logging.info(
      "Execution providers for device '%s': %s",
      device.value,
      execution_providers.select_providers(device),
  )
  repository = models.ModelRepository(args.model_repository, device)
  servicer = server.InferenceServicer(
      repository, allow_shared_memory=args.allow_client_shm
  )

  to_load = []
  if args.model_control_mode in ("none", "poll"):
    to_load = ["*"]
  elif args.load_model:
    to_load = args.load_model
  if "*" in to_load:
    to_load = [s.name for s in repository.index()]
  for name in to_load:
    try:
      repository.load(name)
    except models.ModelError as e:
      logging.error("%s", e)

  grpc_server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=args.max_workers),
      options=_GRPC_OPTIONS,
  )
  service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
      servicer, grpc_server
  )
  endpoint = grpc_endpoint(args.grpc_address, args.grpc_port)
  if grpc_server.add_insecure_port(endpoint) == 0 and not endpoint.startswith(
      "unix:"
  ):
    raise RuntimeError(f"Failed to listen on {endpoint}")
  logging.info("Serving on %s", endpoint)
  return grpc_server, servicer, repository


def main(argv: Sequence[str]) -> None:
  logging.set_verbosity(logging.INFO)
  logging.use_absl_handler()
  # Allow being invoked with Triton's command line, i.e. "tritonserver ...".
  argv = [a for a in argv if a != "tritonserver"]
  args = parse_args(argv)
  if not args.allow_grpc:
    raise SystemExit("Only the gRPC endpoint is supported (--allow-grpc).")
  grpc_server, _, repository = build_server(args)
  grpc_server.start()

  stopped = threading.Event()

  def _stop(signum, frame):
    del frame
    logging.info("Received signal %d, shutting down.", signum)
    stopped.set()

  signal.signal(signal.SIGTERM, _stop)
  signal.signal(signal.SIGINT, _stop)
  stopped.wait()
  grpc_server.stop(grace=5).wait()
  repository.unload_all()


if __name__ == "__main__":
  main(sys.argv[1:])
