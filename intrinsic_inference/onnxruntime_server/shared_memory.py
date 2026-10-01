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

"""Manages POSIX shared memory regions registered by clients.

Implements Triton's system shared memory extension: clients create a region
with shm_open(key), register it by name, and then pass tensors through it by
setting the "shared_memory_region", "shared_memory_offset" and
"shared_memory_byte_size" parameters on inference inputs and outputs.
"""

import dataclasses
import mmap
import os
import threading

# Directory in which shm_open() creates POSIX shared memory objects.
_SHM_DIR = "/dev/shm"


class SharedMemoryError(Exception):
  """An invalid shared memory operation.

  The message always contains "shared memory", which clients use to detect
  shared memory failures and fall back to sending tensors inline.
  """


@dataclasses.dataclass
class Region:
  name: str
  key: str
  offset: int
  byte_size: int
  mapping: mmap.mmap


class SharedMemoryManager:
  """Thread-safe registry of system shared memory regions."""

  def __init__(self, shm_dir: str = _SHM_DIR) -> None:
    self._shm_dir = shm_dir
    self._regions: dict[str, Region] = {}
    self._lock = threading.Lock()

  def register(self, name: str, key: str, offset: int, byte_size: int) -> None:
    """Registers the region `key` under `name`."""
    with self._lock:
      if name in self._regions:
        raise SharedMemoryError(
            f"shared memory region '{name}' already in manager"
        )
      path = os.path.join(self._shm_dir, key.lstrip("/"))
      try:
        fd = os.open(path, os.O_RDWR)
      except OSError as e:
        raise SharedMemoryError(
            f"unable to open shared memory region '{name}' with key '{key}':"
            f" {e}"
        ) from e
      try:
        size = os.fstat(fd).st_size
        if offset < 0 or byte_size <= 0 or offset + byte_size > size:
          raise SharedMemoryError(
              f"invalid offset {offset} and byte size {byte_size} for shared"
              f" memory region '{name}' of size {size}"
          )
        mapping = mmap.mmap(fd, offset + byte_size)
      finally:
        os.close(fd)
      self._regions[name] = Region(name, key, offset, byte_size, mapping)

  def unregister(self, name: str = "") -> None:
    """Unregisters region `name`, or all regions if `name` is empty."""
    with self._lock:
      names = [name] if name else list(self._regions)
      for n in names:
        region = self._regions.pop(n, None)
        if region is not None:
          region.mapping.close()

  def status(self, name: str = "") -> list[Region]:
    """Returns region `name`, or all regions if `name` is empty."""
    with self._lock:
      if not name:
        return list(self._regions.values())
      if name not in self._regions:
        raise SharedMemoryError(
            f"unable to find system shared memory region: '{name}'"
        )
      return [self._regions[name]]

  def _checked_region(self, name: str, offset: int, byte_size: int) -> Region:
    region = self._regions.get(name)
    if region is None:
      raise SharedMemoryError(
          f"unable to find system shared memory region: '{name}'"
      )
    if offset < 0 or byte_size < 0 or offset + byte_size > region.byte_size:
      raise SharedMemoryError(
          f"invalid offset {offset} and byte size {byte_size} for shared"
          f" memory region '{name}' of size {region.byte_size}"
      )
    return region

  def read(self, name: str, offset: int, byte_size: int) -> bytes:
    """Returns `byte_size` bytes at `offset` within region `name`."""
    with self._lock:
      region = self._checked_region(name, offset, byte_size)
      start = region.offset + offset
      return region.mapping[start : start + byte_size]

  def write(self, name: str, offset: int, data: bytes, byte_size: int) -> None:
    """Writes `data` at `offset` within the `byte_size` bytes reserved there."""
    if len(data) > byte_size:
      raise SharedMemoryError(
          f"shared memory size specified with the request ({byte_size}) does"
          f" not match the size of the output ({len(data)})"
      )
    with self._lock:
      region = self._checked_region(name, offset, byte_size)
      start = region.offset + offset
      region.mapping[start : start + len(data)] = data
