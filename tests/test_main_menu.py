import os
import subprocess
import sys
import tempfile
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src import main


class MainMenuItemsTests(unittest.TestCase):
    def test_lightweight_smoke_does_not_require_training_dependencies(self):
        script = """
import builtins

real_import = builtins.__import__

def import_without_training_dependencies(name, *args, **kwargs):
    if name in {"numpy", "torch"} or name.startswith(("numpy.", "torch.")):
        raise ModuleNotFoundError(
            f"{name} is excluded from the lightweight build"
        )
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_training_dependencies
from src import main
main.TORCH_AVAILABLE = False
assert "Train AI" not in [
    label for label, _callback in main.main_menu_items(training_available=False)
]
raise SystemExit(main.package_smoke_test())
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment["SDL_VIDEODRIVER"] = "dummy"
            environment["SDL_AUDIODRIVER"] = "dummy"
            environment["STARAI_DATA_DIR"] = temp_dir
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_ai_is_hidden_when_training_is_unavailable(self):
        labels = [label for label, _ in main.main_menu_items(training_available=False)]

        self.assertNotIn("Train AI", labels)
        self.assertIn("Play Game", labels)

    def test_train_ai_is_shown_when_training_is_available(self):
        labels = [label for label, _ in main.main_menu_items(training_available=True)]

        self.assertIn("Train AI", labels)


if __name__ == "__main__":
    unittest.main()
