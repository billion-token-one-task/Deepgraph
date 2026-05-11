import importlib.util
import unittest
from pathlib import Path

from agents.agent_registry import iter_agent_boundaries


ROOT = Path(__file__).resolve().parents[1]


class AgentArchitectureTests(unittest.TestCase):
    def test_big_agent_folders_exist_and_have_readmes(self):
        for boundary in iter_agent_boundaries():
            folder = ROOT / boundary.folder
            self.assertTrue(folder.is_dir(), boundary.folder)
            self.assertTrue((folder / "__init__.py").is_file(), boundary.folder)
            self.assertTrue((folder / "README.md").is_file(), boundary.folder)

    def test_registry_points_to_importable_modules(self):
        for boundary in iter_agent_boundaries():
            self.assertTrue(boundary.modules, boundary.key)
            for module_name in boundary.modules:
                self.assertIsNotNone(importlib.util.find_spec(module_name), module_name)

    def test_registry_scripts_exist(self):
        for boundary in iter_agent_boundaries():
            for script_name in boundary.scripts:
                relative = Path(*script_name.split(".")).with_suffix(".py")
                path = ROOT / relative
                if not path.is_file():
                    path = (ROOT / Path(*script_name.split("."))).with_suffix(".sh")
                if not path.is_file():
                    path = (ROOT / Path(*script_name.split("."))).with_suffix(".ps1")
                self.assertTrue(path.is_file(), script_name)


if __name__ == "__main__":
    unittest.main()
