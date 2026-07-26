import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.persistence import atomic_write_json
from src.training.model_persistence import (
    resolve_model_save,
    rotate_previous_model_save,
)
from src.training.model_loader import load_inference_model
from src.training.model_registry import (
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    model_architecture_metadata,
    model_paths,
    previous_model_paths,
)
from src.training.replay import save_training_checkpoint
from src.training.value_network import ValueNetworkConfig, build_value_network


def _metadata(completed_batches: int, *, description: str = "Model"):
    return {
        "ship": "Earthling",
        "slot": 1,
        "description": description,
        "architecture": model_architecture_metadata(8, 1),
        "progress": {"completed_batches": completed_batches},
    }


class ModelSaveBackupTests(unittest.TestCase):
    def setUp(self):
        self.model = build_value_network(ValueNetworkConfig(8, 1))

    def _save_pair(self, directory: Path, completed_batches: int):
        checkpoint_path, metadata_path = model_paths(directory, "Earthling", 1)
        save_training_checkpoint(
            checkpoint_path,
            self.model,
            extra_state={"completed_batches": completed_batches},
        )
        atomic_write_json(metadata_path, _metadata(completed_batches))
        return checkpoint_path, metadata_path

    def _slot(self, directory: Path):
        checkpoint_path, metadata_path = model_paths(directory, "Earthling", 1)
        return TrainingModelSlot(
            ship="Earthling",
            slot=1,
            source=SLOT_USER,
            description="Model",
            pth_path=checkpoint_path,
            metadata_path=metadata_path,
            metadata=_metadata(0),
        )

    def test_rotation_keeps_exact_previous_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            expected_checkpoint = checkpoint_path.read_bytes()
            expected_metadata = metadata_path.read_bytes()

            self.assertTrue(
                rotate_previous_model_save(checkpoint_path, metadata_path)
            )

            previous_checkpoint, previous_metadata = previous_model_paths(
                root,
                "Earthling",
                1,
            )
            self.assertEqual(previous_checkpoint.read_bytes(), expected_checkpoint)
            self.assertEqual(previous_metadata.read_bytes(), expected_metadata)

    def test_mismatched_current_pair_resolves_to_matching_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            rotate_previous_model_save(checkpoint_path, metadata_path)
            save_training_checkpoint(
                checkpoint_path,
                self.model,
                extra_state={"completed_batches": 5},
            )

            resolved = resolve_model_save(self._slot(root))
            previous_checkpoint, previous_metadata = previous_model_paths(
                root,
                "Earthling",
                1,
            )

            self.assertEqual(resolved.pth_path, previous_checkpoint)
            self.assertEqual(resolved.metadata_path, previous_metadata)
            self.assertEqual(resolved.metadata["progress"]["completed_batches"], 4)

    def test_matching_current_pair_is_preferred_over_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            rotate_previous_model_save(checkpoint_path, metadata_path)
            self._save_pair(root, 5)

            resolved = resolve_model_save(self._slot(root))

            self.assertEqual(resolved.pth_path, checkpoint_path)
            self.assertEqual(resolved.metadata_path, metadata_path)
            self.assertEqual(resolved.metadata["progress"]["completed_batches"], 5)

    def test_inference_load_falls_back_to_matching_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            rotate_previous_model_save(checkpoint_path, metadata_path)
            save_training_checkpoint(
                checkpoint_path,
                self.model,
                extra_state={"completed_batches": 5},
            )

            loaded = load_inference_model(self._slot(root), device_choice="cpu")
            previous_checkpoint, _ = previous_model_paths(
                root,
                "Earthling",
                1,
            )

            self.assertEqual(loaded.slot.pth_path, previous_checkpoint)
            self.assertEqual(loaded.completed_batches, 4)

    def test_repository_exposes_previous_pair_when_current_pair_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user_dir = root / "user"
            checkpoint_path, metadata_path = self._save_pair(user_dir, 4)
            rotate_previous_model_save(checkpoint_path, metadata_path)
            checkpoint_path.unlink()
            metadata_path.unlink()
            repository = TrainingModelRepository(root / "bundled", user_dir)

            slot = resolve_model_save(repository.slot_for("Earthling", 1))
            previous_checkpoint, previous_metadata = previous_model_paths(
                user_dir,
                "Earthling",
                1,
            )

            self.assertTrue(slot.is_user)
            self.assertEqual(slot.pth_path, previous_checkpoint)
            self.assertEqual(slot.metadata_path, previous_metadata)

    def test_invalid_current_pair_does_not_overwrite_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            rotate_previous_model_save(checkpoint_path, metadata_path)
            previous_checkpoint, previous_metadata = previous_model_paths(
                root,
                "Earthling",
                1,
            )
            expected_checkpoint = previous_checkpoint.read_bytes()
            expected_metadata = previous_metadata.read_bytes()
            save_training_checkpoint(
                checkpoint_path,
                self.model,
                extra_state={"completed_batches": 5},
            )

            self.assertFalse(
                rotate_previous_model_save(checkpoint_path, metadata_path)
            )
            self.assertEqual(previous_checkpoint.read_bytes(), expected_checkpoint)
            self.assertEqual(previous_metadata.read_bytes(), expected_metadata)

    def test_backup_failure_aborts_before_callers_replace_current_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path, metadata_path = self._save_pair(root, 4)
            expected_checkpoint = checkpoint_path.read_bytes()
            expected_metadata = metadata_path.read_bytes()

            with mock.patch(
                "src.training.model_persistence.atomic_copy_file",
                side_effect=[None, OSError("disk error")],
            ):
                with self.assertRaisesRegex(OSError, "disk error"):
                    rotate_previous_model_save(checkpoint_path, metadata_path)

            self.assertEqual(checkpoint_path.read_bytes(), expected_checkpoint)
            self.assertEqual(metadata_path.read_bytes(), expected_metadata)


if __name__ == "__main__":
    unittest.main()
