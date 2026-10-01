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

"""Tests for model_controller_base."""

import threading
import time

from absl.testing import absltest

from intrinsic_inference.core import model_assets_manager_base
from intrinsic_inference.core import model_controller_base
from intrinsic_inference.core.v1 import ml_model_pb2


def _model(name: str, version: str) -> ml_model_pb2.MlModel:
  model = ml_model_pb2.MlModel()
  model.model_config.name = name
  model.model_config.version = version
  return model


class _FakeAssetsManager(model_assets_manager_base.ModelAssetsManagerBase):
  """Keeps one directory per model name, like the real assets manager."""

  def __init__(self):
    self.dirs: dict[str, str] = {}
    self._lock = threading.Lock()

  def list_model_assets(self):
    return {}

  def create_model_asset(self, ml_model):
    time.sleep(0.05)  # Downloading the model files.
    with self._lock:
      self.dirs[ml_model.model_config.name] = ml_model.model_config.version

  def update_model_asset(self, old_model, new_model):
    self.create_model_asset(new_model)

  def delete_model_asset(self, ml_model):
    time.sleep(0.1)  # Unloading from the backend first.
    with self._lock:
      self.dirs.pop(ml_model.model_config.name, None)


class _Controller(model_controller_base.ModelControllerBase):

  def _load_model_impl(self, model_proto):
    pass

  def _unload_model_impl(self, model_proto):
    pass

  def _are_backend_config_equal(self, old_model, new_model):
    return True

  def _reload_model_impl(self, model_proto):
    pass


class ModelControllerBaseTest(absltest.TestCase):

  def test_new_version_replaces_old_without_deleting_its_files(self):
    assets = _FakeAssetsManager()
    controller = _Controller(assets)
    self.addCleanup(controller.stop)
    old = {"model.1": _model("model", "1")}
    new = {"model.2": _model("model", "2")}
    controller._execute_reconciliation(  # pylint: disable=protected-access
        model_controller_base.ReconciliationDiff(
            to_load={"model.2"}, to_unload={"model.1"}, to_reload=set()
        ),
        current_models=old,
        new_models=new,
    )
    controller.wait_for_idle()

    self.assertEqual(assets.dirs, {"model": "2"})
    self.assertEqual(list(controller.models), ["model.2"])

  def test_unrelated_load_and_unload_both_happen(self):
    assets = _FakeAssetsManager()
    assets.dirs["a"] = "1"
    controller = _Controller(assets)
    self.addCleanup(controller.stop)
    controller._execute_reconciliation(  # pylint: disable=protected-access
        model_controller_base.ReconciliationDiff(
            to_load={"b.1"}, to_unload={"a.1"}, to_reload=set()
        ),
        current_models={"a.1": _model("a", "1")},
        new_models={"b.1": _model("b", "1")},
    )
    controller.wait_for_idle()

    self.assertEqual(assets.dirs, {"b": "1"})
    self.assertEqual(list(controller.models), ["b.1"])


if __name__ == "__main__":
  absltest.main()
