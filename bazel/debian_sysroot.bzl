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

"""Repository rule that assembles a C/C++ sysroot from Debian packages."""

_BUILD_FILE = """
filegroup(
    name = "all_files",
    srcs = glob(["lib/**", "usr/include/**", "usr/lib/**"]),
    visibility = ["//visibility:public"],
)
"""

# Debian packages contain absolute symlinks (e.g. usr/lib/<triple>/libm.so ->
# /lib/<triple>/libm.so.6), which would resolve to the build host's libraries.
# Rewrite them relative to the sysroot root.
_RELATIVIZE_SYMLINKS = """
set -euo pipefail
find . -type l -lname '/*' -print0 | while IFS= read -r -d '' link; do
  target="$(readlink "${link}")"
  # "./a/b/link" -> "../../"
  up="$(dirname "${link#./}" | sed -e 's|[^/][^/]*|..|g')"
  ln -sfn "${up}${target}" "${link}"
done
# Drop links into packages that are not part of the sysroot (e.g. libgcc-10-dev's
# libasan.so), which Bazel rejects as inputs.
find . -xtype l -delete
"""

def _debian_sysroot_impl(rctx):
    for i, (url, sha256) in enumerate(rctx.attr.debs.items()):
        deb_dir = "_debs/%d" % i
        rctx.download_and_extract(url = url, sha256 = sha256, type = "deb", output = deb_dir)
        data = [f for f in rctx.path(deb_dir).readdir() if f.basename.startswith("data.tar")]
        if len(data) != 1:
            fail("Expected exactly one data.tar.* in %s" % url)

        # Not rctx.extract(), which mangles the absolute symlinks fixed below.
        result = rctx.execute(["tar", "-xf", data[0], "-C", "."])
        if result.return_code != 0:
            fail("Failed to extract %s: %s" % (url, result.stderr))
    rctx.delete("_debs")

    result = rctx.execute(["bash", "-c", _RELATIVIZE_SYMLINKS])
    if result.return_code != 0:
        fail("Failed to rewrite sysroot symlinks: " + result.stderr)

    rctx.file("BUILD.bazel", _BUILD_FILE)

debian_sysroot = repository_rule(
    implementation = _debian_sysroot_impl,
    attrs = {
        "debs": attr.string_dict(
            mandatory = True,
            doc = "Maps .deb URLs to their SHA-256 checksums.",
        ),
    },
    doc = "Extracts the given Debian packages into a sysroot for toolchains_llvm.",
)
