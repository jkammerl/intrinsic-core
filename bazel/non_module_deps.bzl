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

"""
Module extension for non-module dependencies
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file", "http_jar")
load("//bazel:debian_sysroot.bzl", "debian_sysroot")

# Module extension function signatures include a ctx variable, which should not be removed.
def _non_module_deps_impl(ctx):
    # To update this, see https://github.com/intrinsic-ai/intrinsic-core/blob/main/bazel/sysroot/README.md.  
    http_archive(
        name = "intrinsic_llvm_sysroot",
        sha256 = "24d7e61ceb0a26a2002bd0a3e87dfbc8e12ec95456bd1cced7fc5ccd79c47ed8",
        build_file_content = """
filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"]
)""",
        urls = ["https://storage.googleapis.com/intrinsic-mirror/bazel/sysroot-2025-07-22-845e86b8.tar.zst"],
    )

    # Debian bullseye (glibc 2.31, same as the x86_64 sysroot above) sysroot
    # for linux-aarch64 targets, so that arm64 binaries run in the distroless
    # base images instead of requiring the build host's glibc. Uses the same
    # snapshot as //bazel/debian_snapshot:bullseye_mesa.yaml.
    debian_sysroot(
        name = "intrinsic_llvm_sysroot_aarch64",
        debs = {
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/g/glibc/libc6_2.31-13+deb11u11_arm64.deb": "baaa9aa184e2f21738c5819055e6740cc5b22f198e3f416e33f82b40ff6933d8",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/g/glibc/libc6-dev_2.31-13+deb11u11_arm64.deb": "28d478134722dcd4b0bd2045a199301d18713bf95947b9fce66634e7aeacab2e",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/g/gcc-10/libgcc-10-dev_10.2.1-6_arm64.deb": "80fc73d339ada2e194175afe83cd89014565242153b6e3c2128849d8817da367",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/g/gcc-10/libgcc-s1_10.2.1-6_arm64.deb": "e2fcdb378d3c1ad1bcb64d4fb6b37aab44011152beca12a4944f435a2582df1f",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/l/linux/linux-libc-dev_5.10.223-1_arm64.deb": "8b6374a64412d33eac61d74f77b8f932da4b8a707ea8a614791e2a35b8917618",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/libx/libxcrypt/libcrypt-dev_4.4.18-4_arm64.deb": "5309fdf445acc72794b9d134e9625746ffad2b865c6432abcd08ef097587bde1",
            "https://snapshot.debian.org/archive/debian/20260320T204549Z/pool/main/libx/libxcrypt/libcrypt1_4.4.18-4_arm64.deb": "22b586b29e840dabebf0bf227d233376628b87954915d064bc142ae85d1b7979",
        },
    )




    # Download and extract GLVND EGL/GL stubs from the sysroot for the host
    # architecture. Needed for tests that require OpenGL/EGL
    if ctx.os.arch in ("aarch64", "arm64"):
        # Debian bullseye arm64 sysroot from Chromium's
        # build/linux/sysroot_scripts/sysroots.json.
        glvnd_stubs_url = "https://commondatastorage.googleapis.com/chrome-linux-sysroot/c7176a4c7aacbf46bda58a029f39f79a68008d3dee6518f154dcf5161a5486d8"
        glvnd_stubs_sha256 = "c7176a4c7aacbf46bda58a029f39f79a68008d3dee6518f154dcf5161a5486d8"
    else:
        glvnd_stubs_url = "https://storage.googleapis.com/chrome-linux-sysroot/toolchain/4f611ec025be98214164d4bf9fbe8843f58533f7/debian_bullseye_amd64_sysroot.tar.xz"
        glvnd_stubs_sha256 = "5df5be9357b425cdd70d92d4697d07e7d55d7a923f037c22dc80a78e85842d2c"
    http_archive(
        name = "sysroot_glvnd_stubs",
        build_file = Label("//intrinsic/production/external:sysroot_glvnd_stubs.BUILD.bazel"),
        urls = [glvnd_stubs_url],
        sha256 = glvnd_stubs_sha256,
        type = "tar.xz",
    )

    http_jar(
        name = "firestore_emulator",
        sha256 = "9d43599ed6151199e8d604dc87fac51218e49e5f3a48519b1ae560bbe5e3382d",
        urls = ["https://storage.googleapis.com/firebase-preview-drop/emulator/cloud-firestore-emulator-v1.19.8.jar"],
    )

    http_archive(
        name = "coal",
        build_file = Label("//intrinsic/production/external:coal.BUILD.bazel"),
        sha256 = "136d7f809fccd0027265dead58abb57798fc7631f1a3331f139a74bed92066cc",
        strip_prefix = "coal-1.0.1",
        urls = ["https://github.com/torresl-google/coal/archive/refs/tags/v1.0.1.tar.gz"],
    )

    http_archive(
        name = "cgal",
        build_file = Label("//intrinsic/production/external:cgal.BUILD.bazel"),
        sha256 = "4a0713351882989dda8b6bbfa942d5013b994f297fbebe1096ce664a92aff1a0",
        strip_prefix = "cgal-6.2.1",
        url = "https://github.com/CGAL/cgal/archive/refs/tags/v6.2.1.tar.gz",
    )

    # For ur_calibration.[cc/h]
    http_archive(
        name = "ur_calibration",
        build_file = Label("//intrinsic/production/external:ur_calibration.BUILD.bazel"),
        patch_args = ["-p1"],
        patches = [
            # TODO(b/382000146): Remove patch once Logger can be selected.
            Label("//intrinsic/production/external/patches:ur-calibration.patch"),
        ],
        sha256 = "e3b5a7b4c1d9ea2e59497b1c5f029562d8352828d3213d9bac46a555fbdf6786",
        strip_prefix = "Universal_Robots_ROS2_Driver-4.1.0/ur_calibration",
        urls = ["https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/archive/refs/tags/4.1.0.tar.gz"],
    )

    # TODO build kubectl from source (go package "k8s.io/kubernetes/cmd/kubectl")
    # - https://github.com/bazel-contrib/bazel-gazelle/issues/1392#issuecomment-2642983316
    http_archive(
        name = "kubebuilder_test_tools",
        build_file_content = """
exports_files(
    [
        "etcd",
        "kube-apiserver",
        "kubectl",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "tools",
    srcs = glob(["*"]),
    visibility = ["//visibility:public"],
)
""",
        sha256 = "2a9792cb5f1403f524543ce94c3115e3c4a4229f0e86af55fd26c078da448164",
        type = "tar.gz",
        urls = ["https://github.com/kubernetes-sigs/controller-tools/releases/download/envtest-v1.30.0/envtest-v1.30.0-linux-amd64.tar.gz"],
        strip_prefix = "controller-tools/envtest",
    )

    # Schema JSON files for helm chart validation.
    # Generated by:
    #
    # git clone https://github.com/yannh/kubernetes-json-schema
    # cd kubernetes-json-schema
    # tar zcf schemas.tar.gz master-standalone-strict
    # gcloud storage mv schemas.tar.gz gs://intrinsic-mirror/bazel/helmlint/schemas-$(git rev-parse HEAD).tar.gz
    #
    # We do not depend on the full git repository as a http_arcive, because the archive is huge
    # (~1GB) and this file we need is only a few MB.
    http_file(
        name = "kubernetes_json_schema",
        sha256 = "7df8fe4a1739b56b03bb3794f64a77bcf978d9de11548ed920526e090864b9ed",
        downloaded_file_path = "schemas.tar.gz",
        urls = ["https://storage.googleapis.com/intrinsic-mirror/bazel/helmlint/schemas-b3d60bb6a5c10815dcfb1ee592b4e5b6f10cef29.tar.gz"],
    )

    PERFKITBENCHMARKER_COMMIT = "def24810d209eaa38abb802547edb1e144d515da"  # 2024-22-10
    http_file(
        name = "com_googlecloudplatform_perfkitbenchmarker_tests_matchers_py",
        downloaded_file_path = "matchers.py",
        urls = ["https://raw.githubusercontent.com/GoogleCloudPlatform/PerfKitBenchmarker/%s/tests/matchers.py" % PERFKITBENCHMARKER_COMMIT],
        sha256 = "1e5045af622e088d711ceb3dbda96b7c4ce19356b5335860862b82357eb35eec",
    )

    S2GEOMETRY_COMMIT = "418c55893f6123b90f2768ed2ec9f5f47fa512de"  # 2024-05-12
    http_file(
        name = "com_google_s2geometry_bit_interleave_h",
        downloaded_file_path = "bit-interleave.h",
        urls = ["https://raw.githubusercontent.com/google/s2geometry/%s/src/s2/util/bits/bit-interleave.h" % S2GEOMETRY_COMMIT],
        sha256 = "9090fb46e735b396cbda71546d7fdd2d66fb06bc224233ac05db2c8d58c9bf0a",
    )
    http_file(
        name = "com_google_s2geometry_bit_interleave_cc",
        downloaded_file_path = "bit-interleave.cc",
        urls = ["https://raw.githubusercontent.com/google/s2geometry/%s/src/s2/util/bits/bit-interleave.cc" % S2GEOMETRY_COMMIT],
        sha256 = "44fa1fb4f8db3f305f08ddf46367f93e1c3b1364779e4bf888771de11447d8c0",
    )

    GLOG_COMMIT = "7b134a5c82c0c0b5698bb6bf7a835b230c5638e4"  # v0.7.1
    http_file(
        name = "com_google_glog_googleinit_h",
        downloaded_file_path = "googleinit.h",
        urls = ["https://raw.githubusercontent.com/google/glog/%s/src/base/googleinit.h" % GLOG_COMMIT],
        sha256 = "ee770ce4721fe37dd3ef6a80e4f646082d835bf4923d235c29484641b0129782",
    )

    http_file(
        name = "oss_licenses",
        downloaded_file_path = "oss_licenses.json",
        urls = ["https://storage.googleapis.com/intrinsic-mirror/oss_licenses/intrinsic.platform.20251110.RC08.json"],
        # This is a load bearing comment.
        sha256 = "7022570bc17af774bdf42c5214e3b270c9250ace8f3df1d6377583c5efb785df",  # OSS_LICENSES_SHA256
    )

    http_file(
        name = "libkmsp11",
        downloaded_file_path = "libkmsp11-1.1-linux-amd64.tar.gz",
        urls = ["https://github.com/GoogleCloudPlatform/kms-integrations/releases/download/v1.1/libkmsp11-1.1-linux-amd64.tar.gz"],
        sha256 = "d75c991cbf3fa391b1b95b95358feaa41894a608c8ded21028cdfa6fc5f0c378",
    )



non_module_deps_ext = module_extension(
    implementation = _non_module_deps_impl,
    # sysroot_glvnd_stubs depends on the host architecture.
    arch_dependent = True,
)
