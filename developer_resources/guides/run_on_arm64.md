# Run Intrinsic Core on arm64

Intrinsic Core builds and runs natively on arm64 (aarch64) Linux hosts, for
example Ampere servers, NVIDIA DGX Spark or Jetson Thor, and Linux VMs on Apple
silicon Macs (UTM, Parallels, VMware Fusion). This guide covers what works, how
to install Intrinsic Core on arm64, and how to port a Solution such as the
[Open Machine Tending Solution (OMTS)](https://github.com/intrinsic-ai/intrinsic-omts)
to arm64.

![OMTS running natively on arm64 in Gazebo](../img/guides/omts_arm64_gazebo.gif)

_A machine-tending cycle of OMTS running natively on an arm64 VM without a
GPU, shown in the Gazebo GUI at 2x speed. Pose estimation runs on the CPU
before the robot moves; that part is cut._

> [!NOTE]
> arm64 support is new. Release artifacts (`intrinsic-base`, `inctl`) are
> only published for x86-64, so on arm64 you build them from source. x86-64
> remains the recommended platform for real-time control of robot hardware.

## Status

| Component | arm64 status |
| :--- | :--- |
| Intrinsic Core runtime (`intrinsic-base`) | Builds from source and runs natively. |
| `inctl` | Builds from source. |
| Container images built with `python_oci_image` and `container_image` | Built for the host architecture. Base images must be multi-arch (see [Port a Solution](#port-a-solution-to-arm64)). |
| ML inference service | Runs models with the [ONNX Runtime server](../../intrinsic_inference/onnxruntime_server/README.md) on the CPU, or on NVIDIA GPUs of arm64 hosts with CUDA. Triton Inference Server is x86-64 only. |
| RViz | Works with the arm64 ROS 2 packages (`ros-lyrical-desktop`). |
| Gazebo simulation | Runs natively, including the Gazebo GUI (`gz sim -g` from the ROS 2 Lyrical packages). |
| Real-time control (ICON on `PREEMPT_RT`) | Not verified on arm64. Use an x86-64 PC for real robots. |
| OMTS | Runs end to end in simulation (pick, place, unload, return); see [Port a Solution](#port-a-solution-to-arm64). |

### GPUs

* **NVIDIA arm64 hosts** (DGX Spark, Jetson Thor, Grace): run
  `setup_nvidia.sh` as on x86-64. The inference service uses CUDA
  automatically.
* **Hosts without an NVIDIA GPU**, including VMs on Apple silicon: models run
  on the CPU. That works but is slow; for example, FoundationPose pose
  estimation takes about 75 seconds per pose on 6 cores, compared to well
  under a second on an RTX GPU. Don't run `setup_nvidia.sh`: it stops with an
  error when no NVIDIA GPU is present.
* **VMs on Apple silicon** can't use the Mac's GPU for compute: macOS doesn't
  pass it through to Linux guests, and it doesn't support CUDA.

## Install Intrinsic Core on arm64

Follow [Getting started](../learn/tutorials/getting_started.md) on Ubuntu 26.04
for arm64, with these differences:

1. Install Bazelisk first, using the arm64 build. You need it to build
   `intrinsic-base` and `inctl` in the next steps:

   ```bash
   curl -L "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64" -o /tmp/bazelisk
   sudo mv /tmp/bazelisk /usr/bin/bazelisk
   sudo chmod +x /usr/bin/bazelisk
   sudo ln -sf /usr/bin/bazelisk /usr/bin/bazel
   ```

2. `setup_k3s.sh` installs the arm64 builds of k3s, k9s, helm and istioctl
   automatically.

3. Instead of downloading the `intrinsic-base` release, build and deploy it
   from source. The first build can take an hour or more:

   ```bash
   cd ~/intrinsic-core
   bazel run //intrinsic_runtime/kubernetes/intrinsic_base:intrinsic-base
   ```

   On machines with 16 GiB of RAM, limit the build's parallelism, for example
   by adding `--jobs=3 --local_resources=memory=HOST_RAM*.6` after `run`.

4. Instead of downloading the `inctl` release, build it from source:

   ```bash
   cd ~/intrinsic-core
   bazel build //intrinsic/tools/inctl:inctl_external
   sudo install -m 755 bazel-bin/intrinsic/tools/inctl/inctl_external /usr/local/bin/inctl
   ```

5. Only run `setup_nvidia.sh` on hosts with an NVIDIA GPU.

6. The LLVM compatibility symlink uses the arm64 library directory:

   ```bash
   sudo ln -s /usr/lib/aarch64-linux-gnu/libxml2.so.16 /usr/lib/aarch64-linux-gnu/libxml2.so.2
   ```

## How arm64 builds work

You don't need to pass any extra flags: `bazel` commands work the same on both
architectures.

* **`tools/bazel`**: Bazelisk runs this wrapper instead of Bazel. On arm64
  hosts it adds `--config=arm64`; on x86-64 hosts it does nothing.
* **`--config=arm64`** (in `.bazelrc`): sets `--cpu=aarch64`, overriding the
  default `--config=haswell`, and selects the `//bazel:linux_aarch64` host
  platform.
* **Sysroot**: C++ code links against a glibc 2.31 arm64 sysroot, assembled
  from pinned Debian bullseye packages (`@intrinsic_llvm_sysroot_aarch64`), so
  binaries run in the distroless base images just like x86-64 binaries.
* **Python**: pip dependencies are resolved for `linux_aarch64` and
  `linux_x86_64`.
* **Container images**: images are built for the host's architecture. Base
  images are pulled for both `linux/amd64` and `linux/arm64`, where the
  upstream image supports it.

Builds for x86-64 are unchanged. Cross-compiling from one architecture to the
other is not supported; build on the architecture you deploy to.

## Port a Solution to arm64

A Solution repository, such as OMTS or your own, depends on Intrinsic Core as
a Bazel module. Bazel's arm64 settings from Intrinsic Core's `.bazelrc`,
`tools/bazel` and root `MODULE.bazel` don't carry over to modules that depend
on it, so a Solution needs the following changes of its own.

### 1. Apply `--config=arm64` automatically

Copy [`tools/bazel`](../../tools/bazel) into your repository at the same path
and make it executable. Then add to your `.bazelrc`:

```text
# --config arm64 for building on arm64 hosts; tools/bazel applies it.
common:arm64 --cpu=aarch64
common:arm64 --host_platform=@intrinsic-core//bazel:linux_aarch64
```

If your `.bazelrc` sets x86-only compiler flags (such as `-mcrc32`,
`-march=haswell` or `--cpu=haswell`), move them to a `common:haswell` config
or into the LLVM toolchain's `linux-x86_64` flags.

### 2. Add the arm64 sysroot to the LLVM toolchain

In `MODULE.bazel`, next to the existing x86-64 sysroot:

```starlark
non_module_deps = use_extension("@intrinsic-core//bazel:non_module_deps.bzl", "non_module_deps_ext")
use_repo(
    non_module_deps,
    "intrinsic_llvm_sysroot",
    "intrinsic_llvm_sysroot_aarch64",
)

llvm.sysroot(
    label = "@intrinsic_llvm_sysroot//:all_files",
    targets = ["linux-x86_64"],
)

llvm.sysroot(
    label = "@intrinsic_llvm_sysroot_aarch64//:all_files",
    targets = ["linux-aarch64"],
)
```

Without this, arm64 binaries link against the build host's glibc and fail at
runtime with errors like `GLIBC_2.42 not found`.

### 3. Resolve pip dependencies for arm64

Add `linux_aarch64` to every `pip.parse`:

```starlark
pip.parse(
    ...
    target_platforms = [
        "linux_aarch64",
        "linux_x86_64",
    ],
)
```

Then check that every locked package has an aarch64 wheel or an sdist.
`manylinux_*_aarch64` wheels exist for most packages, including numpy,
opencv-python-headless, onnxruntime and onnxruntime-gpu.

### 4. Pull multi-arch base images

For each `oci.pull` whose image is published for arm64, list both platforms:

```starlark
oci.pull(
    name = "my_base",
    digest = "sha256:...",  # The digest of the multi-arch index.
    image = "...",
    platforms = [
        "linux/amd64",
        "linux/arm64/v8",
    ],
)
```

Images that are only published for amd64 can't be used on arm64; replace them
or make the targets that use them x86-64 only.

### 5. Handle x86-only and CUDA-only code

* **ML models**: serve them through the inference service. Its ONNX Runtime
  server loads Triton model repositories (the `onnxruntime` and `python`
  backends) and runs them on the CPU or on CUDA. Python backend models can't
  rely on Triton's `EXECUTION_ENV_PATH` environments: import only packages
  that the server image contains (numpy, OpenCV, trimesh, onnxruntime), and
  ship any other Python modules next to `model.py`.
* **CUDA code** (for example `rules_cuda` targets): provide a CPU
  implementation and fall back to it when CUDA isn't available, or restrict
  the target to x86-64 with
  `target_compatible_with = ["@platforms//cpu:x86_64"]`.
* **Architecture-specific paths**: use `$(uname -m)` in scripts instead of
  hard-coding `x86_64`, `amd64` or `x86_64-linux-gnu`.

### 6. Verify

On an arm64 host:

```bash
bazel build //...
bazel test //...
```

Then deploy the Solution in simulation and check that no pod fails with
`exec format error`, which means that it's running an image built for the
other architecture.

## Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| `exec format error` or `Exec format error` when running a binary or starting a pod | The binary or image was built for x86-64, e.g. a downloaded release artifact. | Build it from source on the arm64 host, as described in [Install Intrinsic Core on arm64](#install-intrinsic-core-on-arm64). |
| `version 'GLIBC_2.xx' not found` in a container | The binary was linked against the build host's glibc instead of the sysroot. | Add the arm64 sysroot to the LLVM toolchain (see [step 2](#2-add-the-arm64-sysroot-to-the-llvm-toolchain)). |
| `could not find an image matching the target platform` | An `oci.pull` base image doesn't list `linux/arm64`. | Add the arm64 platform (see [step 4](#4-pull-multi-arch-base-images)), or replace the image if it isn't published for arm64. |
| Compiler errors about `-mcrc32`, `-march=haswell` or x86 intrinsics | `--config=arm64` isn't applied, or the repository sets x86-only flags. | Check that `tools/bazel` exists and is executable, and that you run `bazel` through Bazelisk (see [step 1](#1-apply---configarm64-automatically)). |
| Pose estimation takes more than a minute | Models run on the CPU because there is no NVIDIA GPU. | Expected without a GPU. For faster inference, use an arm64 host with an NVIDIA GPU. |
