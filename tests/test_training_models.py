import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_registry import (
    SLOT_BUNDLED,
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelRepository,
    metadata_from_state,
    model_paths,
)


class TrainingModelRepositoryTests(unittest.TestCase):
    def test_empty_slot_is_available_without_bundled_or_user_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")

            slot = repository.slot_for("Arilou", 1)

        self.assertEqual(slot.source, SLOT_EMPTY)
        self.assertFalse(slot.exists)

    def test_bundled_model_is_read_only_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled_pth, bundled_json = model_paths(root / "bundled", "Arilou", 1)
            bundled_pth.parent.mkdir()
            bundled_pth.touch()
            bundled_json.write_text('{"description": "Optimized"}', encoding="utf-8")
            repository = TrainingModelRepository(root / "bundled", root / "user")

            slot = repository.slot_for("Arilou", 1)

        self.assertEqual(slot.source, SLOT_BUNDLED)
        self.assertEqual(slot.description, "Default")
        self.assertTrue(slot.is_bundled)

    def test_user_model_description_comes_from_sidecar_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Mycon",
                slot=2,
                description="Aggressive",
                architecture={"hidden_layer_size": 128},
                training={"regimen": {"rounds_per_epoch": 10}},
            )

            repository.create_or_update_user_model(metadata)
            slot = repository.slot_for("Mycon", 2)
            pth_path, metadata_path = model_paths(root / "user", "Mycon", 2)

            self.assertEqual(slot.source, SLOT_USER)
            self.assertEqual(slot.description, "Aggressive")
            self.assertEqual(pth_path.read_bytes(), b"")
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                metadata,
            )

    def test_pth_without_json_is_existing_model_with_empty_description(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pth_path, _ = model_paths(root / "user", "Earthling", 3)
            pth_path.parent.mkdir()
            pth_path.touch()
            repository = TrainingModelRepository(root / "bundled", root / "user")

            slot = repository.slot_for("Earthling", 3)

        self.assertEqual(slot.source, SLOT_USER)
        self.assertEqual(slot.description, "")


if __name__ == "__main__":
    unittest.main()
