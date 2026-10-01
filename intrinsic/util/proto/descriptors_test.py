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

"""Tests for descriptors.py."""

from absl.testing import absltest
from google.protobuf import api_pb2
from google.protobuf import descriptor_pool
from google.protobuf import type_pb2

from intrinsic.util.proto import descriptors


class GenFileDescriptorSetTest(absltest.TestCase):

  def test_contains_transitive_dependencies(self):
    fds = descriptors.gen_file_descriptor_set(api_pb2.Api.DESCRIPTOR)

    self.assertCountEqual(
        [f.name for f in fds.file],
        [
            "google/protobuf/any.proto",
            "google/protobuf/api.proto",
            "google/protobuf/source_context.proto",
            "google/protobuf/type.proto",
        ],
    )

  def test_lists_dependencies_before_dependents(self):
    fds = descriptors.gen_file_descriptor_set(api_pb2.Api.DESCRIPTOR)

    # A fresh pool only accepts a file once all of its imports are present,
    # like Go's protodesc.NewFile with an incrementally filled registry.
    pool = descriptor_pool.DescriptorPool()
    for file_proto in fds.file:
      pool.Add(file_proto)
    self.assertEqual(
        pool.FindMessageTypeByName("google.protobuf.Api").full_name,
        "google.protobuf.Api",
    )

  def test_multiple_messages_have_no_duplicate_files(self):
    fds = descriptors.gen_file_descriptor_set(
        [api_pb2.Api.DESCRIPTOR, type_pb2.Type.DESCRIPTOR]
    )

    names = [f.name for f in fds.file]
    self.assertLen(names, len(set(names)))


if __name__ == "__main__":
  absltest.main()
