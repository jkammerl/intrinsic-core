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

load("//bazel:container.bzl", "container_image")

# Zenoh router image for the intrinsic-base chart, built for the target CPU from
# the upstream release, so it runs natively on amd64 and arm64 hosts.
#
# The files sit in / like in the previously used mirror image: zenohd searches
# its working directory for plugins, and the config is the image's default.

_ZENOHD_FILES = [
    "zenohd",
    "libzenoh_plugin_rest.so",
    "libzenoh_plugin_storage_manager.so",
]

container_image(
    name = "zenohd-image",
    base = "//intrinsic/kubernetes:base-image-cc-oci",
    entrypoint = [
        "/dumb-init",
        "--",
        "/zenohd",
        "-c",
        "/zenohd_config.json5",
    ],
    files = select({
        "@platforms//cpu:aarch64": ["@zenohd_aarch64//:" + f for f in _ZENOHD_FILES] + [
            "@zenoh_backend_filesystem_aarch64//:libzenoh_backend_fs.so",
        ],
        "@platforms//cpu:x86_64": ["@zenohd_x86_64//:" + f for f in _ZENOHD_FILES] + [
            "@zenoh_backend_filesystem_x86_64//:libzenoh_backend_fs.so",
        ],
    }) + ["zenohd_config.json5"],
    visibility = ["//intrinsic_runtime/kubernetes/intrinsic_base:__pkg__"],
    workdir = "/",
)
