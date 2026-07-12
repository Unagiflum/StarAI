import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from src.training.contracts import (
    ACTION_INDEX_TABLE,
    ACTION_OUTPUT_SIZE,
    ACTION_SCHEMA_METADATA,
    OBSERVATION_INPUT_SIZE,
    SHIP_TYPE_CATALOG_ORDER,
)
from src.training.model_registry import (
    MODEL_METADATA_VERSION,
    SLOT_BUNDLED,
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
    model_paths,
    replay_checkpoint_path,
    trained_model_counts_for_ships,
)
from src.training.opponent_cache import (
    ModelSaveCoordinator,
    OpponentModelCache,
    OpponentModelKey,
)
from src.training import torch_backend
from src.training.replay import save_training_checkpoint
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
    predict_action_values,
    selected_action_regression_loss,
)


class TrainingContractTests(unittest.TestCase):
    def test_action_table_has_exactly_24_unique_valid_actions(self):
        masks = [action.mask for action in ACTION_INDEX_TABLE]

        self.assertEqual(len(ACTION_INDEX_TABLE), ACTION_OUTPUT_SIZE)
        self.assertEqual(len(set(masks)), ACTION_OUTPUT_SIZE)
        self.assertEqual(masks, [entry["mask"] for entry in ACTION_SCHEMA_METADATA])
        for action in ACTION_INDEX_TABLE:
            with self.subTest(mask=action.mask):
                self.assertFalse(action.turn_left and action.turn_right)

    def test_action_table_uses_stable_bitmask_order(self):
        self.assertEqual(ACTION_INDEX_TABLE[0].held_controls, ())
        self.assertEqual(ACTION_INDEX_TABLE[1].held_controls, ("thrust",))
        self.assertEqual(ACTION_INDEX_TABLE[2].held_controls, ("turn_left",))
        self.assertEqual(ACTION_INDEX_TABLE[6].held_controls, ("a1",))
        self.assertEqual(
            ACTION_INDEX_TABLE[-1].held_controls,
            ("thrust", "turn_right", "a1", "a2"),
        )

    def test_ship_catalog_order_is_versioned_and_has_25_entries(self):
        self.assertEqual(len(SHIP_TYPE_CATALOG_ORDER), 25)
        self.assertEqual(SHIP_TYPE_CATALOG_ORDER[0], "Androsynth")


class ValueNetworkTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_model_accepts_533_inputs_and_returns_24_raw_outputs(self):
        torch = torch_backend.require_torch()
        config = ValueNetworkConfig(hidden_layer_width=16, hidden_layer_count=2)
        model = build_value_network(config)
        with torch.no_grad():
            final_layer = model[-1]
            final_layer.weight.zero_()
            final_layer.bias.fill_(-0.5)

        output = predict_action_values(model, torch.zeros((3, OBSERVATION_INPUT_SIZE)))

        self.assertEqual(tuple(output.shape), (3, ACTION_OUTPUT_SIZE))
        self.assertTrue(torch.all(output < 0))
        self.assertFalse(torch.allclose(output.sum(dim=1), torch.ones(3)))

    def test_selected_action_regression_targets_only_selected_output_rows(self):
        torch = torch_backend.require_torch()
        model = build_value_network(ValueNetworkConfig(8, 1))
        observations = torch.ones((2, OBSERVATION_INPUT_SIZE))
        actions = torch.tensor([1, 3])
        returns = torch.tensor([2.0, -1.0])

        loss = selected_action_regression_loss(model, observations, actions, returns)
        loss.backward()
        final_weight_grad = model[-1].weight.grad

        self.assertTrue(torch.any(final_weight_grad[1] != 0))
        self.assertTrue(torch.any(final_weight_grad[3] != 0))
        self.assertTrue(torch.all(final_weight_grad[0] == 0))

    def test_optimizer_step_returns_float_loss(self):
        torch = torch_backend.require_torch()
        model = build_value_network(ValueNetworkConfig(8, 1))
        optimizer = build_optimizer(model, learning_rate=0.001)

        loss = selected_action_regression_loss(
            model,
            torch.zeros((1, OBSERVATION_INPUT_SIZE)),
            [0],
            [1.0],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertIsInstance(float(loss.detach().cpu().item()), float)

    def test_model_construction_fails_clearly_without_torch(self):
        with mock.patch("src.training.torch_backend._torch", None):
            with self.assertRaises(torch_backend.TorchUnavailableError):
                build_value_network(ValueNetworkConfig(8, 1))


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
                architecture=model_architecture_metadata(128, 2),
                training={"regimen": {"rounds_per_batch": 10}},
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

    def test_new_user_model_requires_description(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Mycon",
                slot=2,
                description="  ",
                architecture=model_architecture_metadata(128, 2),
                training={"regimen": {"rounds_per_batch": 10}},
            )

            with self.assertRaisesRegex(ValueError, "require a description"):
                repository.create_or_update_user_model(metadata)

    def test_delete_user_model_removes_replay_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Mycon",
                slot=2,
                description="Aggressive",
                architecture=model_architecture_metadata(128, 2),
                training={"regimen": {"rounds_per_batch": 10}},
            )
            repository.create_or_update_user_model(metadata)
            pth_path, metadata_path = model_paths(root / "user", "Mycon", 2)
            pth_path.write_bytes(b"checkpoint")
            replay_path = replay_checkpoint_path(pth_path)
            replay_path.write_bytes(b"replay")

            repository.delete_user_model("Mycon", 2)

        self.assertFalse(pth_path.exists())
        self.assertFalse(metadata_path.exists())
        self.assertFalse(replay_path.exists())

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

    def test_trained_model_counts_include_only_non_empty_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundled_pth, _ = model_paths(root / "bundled", "Earthling", 2)
            bundled_pth.parent.mkdir()
            bundled_pth.write_bytes(b"default checkpoint")
            user_pth, _ = model_paths(root / "user", "Earthling", 1)
            user_pth.parent.mkdir()
            user_pth.write_bytes(b"user checkpoint")
            empty_pth, _ = model_paths(root / "user", "Earthling", 3)
            empty_pth.touch()
            repository = TrainingModelRepository(root / "bundled", root / "user")

            counts = trained_model_counts_for_ships(
                repository,
                ("Earthling", "Mycon"),
            )

        self.assertEqual(counts["Earthling"], 2)
        self.assertEqual(counts["Mycon"], 0)

    def test_metadata_round_trips_training_schema_and_architecture_contracts(self):
        metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="Baseline",
            architecture={"hidden_layer_size": 64, "hidden_layer_count": 4},
            training={"regimen": {"learning_rate": 0.001}},
            game_settings={
                "ship_directions": 32,
                "asteroid_count": 7,
                "repeat_key_delay": 2,
                "fps": 24,
            },
            progress={"completed_batches": 12},
        )

        self.assertEqual(metadata["schema_version"], MODEL_METADATA_VERSION)
        self.assertEqual(metadata["observation_input_size"], OBSERVATION_INPUT_SIZE)
        self.assertEqual(metadata["action_ordering"], [dict(action) for action in ACTION_SCHEMA_METADATA])
        self.assertEqual(metadata["architecture"]["hidden_layer_width"], 64)
        self.assertEqual(metadata["architecture"]["hidden_layer_count"], 4)
        self.assertEqual(metadata["architecture"]["output_count"], ACTION_OUTPUT_SIZE)
        self.assertEqual(metadata["game_settings"]["ship_directions"], 32)
        self.assertEqual(metadata["progress"]["completed_batches"], 12)


class OpponentModelCacheTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_initial_load_reuses_one_shared_model_per_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=7)
            cache = OpponentModelCache()

            cache.load_initial(repository)
            first_snapshot = cache.snapshot()
            cache.load_initial(repository)
            second_snapshot = cache.snapshot()

        self.assertEqual(len(first_snapshot), 1)
        self.assertEqual(len(second_snapshot), 1)
        self.assertEqual(first_snapshot[0].ship, "Earthling")
        self.assertEqual(first_snapshot[0].slot, 1)
        self.assertIs(first_snapshot[0].model, second_snapshot[0].model)
        diagnostics = cache.diagnostics()
        self.assertEqual(
            diagnostics.loaded_keys,
            (OpponentModelKey("Earthling", 1),),
        )
        self.assertEqual(diagnostics.last_errors, {})
        loaded_entry = diagnostics.entries[OpponentModelKey("Earthling", 1)]
        self.assertEqual(loaded_entry.completed_batches, 7)
        self.assertGreater(loaded_entry.checkpoint_size, 0)
        self.assertIsNotNone(loaded_entry.checkpoint_mtime_ns)
        self.assertIs(loaded_entry.model, first_snapshot[0].model)

    def test_snapshot_is_immutable_and_stable_after_later_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=1)
            cache = OpponentModelCache()
            cache.load_initial(repository)
            first_snapshot = cache.snapshot()

            self._save_model(repository, "Mycon", 2, completed_batches=2)
            cache.load_initial(repository)
            second_snapshot = cache.snapshot()

        self.assertIsInstance(first_snapshot, tuple)
        self.assertEqual(
            [(opponent.ship, opponent.slot) for opponent in first_snapshot],
            [("Earthling", 1)],
        )
        self.assertEqual(
            [(opponent.ship, opponent.slot) for opponent in second_snapshot],
            [("Earthling", 1), ("Mycon", 2)],
        )
        self.assertIs(first_snapshot[0].model, second_snapshot[0].model)

    def test_initial_load_records_failed_load_diagnostics_without_snapshot_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            slot = self._create_metadata_only_slot(
                repository,
                "Earthling",
                1,
                completed_batches=1,
            )
            slot.pth_path.write_bytes(b"not a checkpoint")
            cache = OpponentModelCache()

            cache.load_initial(repository)
            snapshot = cache.snapshot()
            diagnostics = cache.diagnostics()

        key = OpponentModelKey("Earthling", 1)
        self.assertEqual(snapshot, ())
        self.assertEqual(diagnostics.loaded_keys, ())
        self.assertIn(key, diagnostics.last_errors)
        self.assertNotIn(key, diagnostics.entries)

    def test_notify_model_saved_refreshes_cached_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=1)
            cache = OpponentModelCache()
            cache.load_initial(repository)
            old_model = cache.snapshot()[0].model

            self._save_model(repository, "Earthling", 1, completed_batches=2)
            cache.notify_model_saved(repository, "Earthling", 1)
            snapshot = cache.snapshot()
            diagnostics = cache.diagnostics()

        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].slot, 1)
        self.assertIsNot(snapshot[0].model, old_model)
        self.assertEqual(diagnostics.last_errors, {})
        self.assertEqual(
            diagnostics.entries[OpponentModelKey("Earthling", 1)].completed_batches,
            2,
        )

    def test_notify_model_saved_keeps_last_good_when_refresh_load_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=1)
            cache = OpponentModelCache()
            cache.load_initial(repository)
            old_model = cache.snapshot()[0].model

            slot = self._create_metadata_only_slot(
                repository,
                "Earthling",
                1,
                completed_batches=2,
            )
            slot.pth_path.write_bytes(b"not a checkpoint")
            cache.notify_model_saved(repository, "Earthling", 1)
            snapshot = cache.snapshot()
            diagnostics = cache.diagnostics()

        key = OpponentModelKey("Earthling", 1)
        self.assertEqual(len(snapshot), 1)
        self.assertIs(snapshot[0].model, old_model)
        self.assertIn(key, diagnostics.last_errors)
        self.assertEqual(diagnostics.entries[key].completed_batches, 1)
        self.assertIs(diagnostics.entries[key].model, old_model)

    def test_notify_model_saved_leaves_missing_key_absent_when_load_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            slot = self._create_metadata_only_slot(
                repository,
                "Earthling",
                1,
                completed_batches=1,
            )
            slot.pth_path.write_bytes(b"not a checkpoint")
            cache = OpponentModelCache()

            cache.notify_model_saved(repository, "Earthling", 1)
            snapshot = cache.snapshot()
            diagnostics = cache.diagnostics()

        key = OpponentModelKey("Earthling", 1)
        self.assertEqual(snapshot, ())
        self.assertEqual(diagnostics.loaded_keys, ())
        self.assertIn(key, diagnostics.last_errors)
        self.assertNotIn(key, diagnostics.entries)

    def test_notify_model_saved_skips_active_save_and_keeps_old_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=1)
            coordinator = ModelSaveCoordinator()
            cache = OpponentModelCache(save_coordinator=coordinator)
            cache.load_initial(repository)
            old_model = cache.snapshot()[0].model
            key = OpponentModelKey("Earthling", 1)

            self._save_model(repository, "Earthling", 1, completed_batches=2)
            with coordinator.saving(key):
                cache.notify_model_saved(repository, "Earthling", 1)
                blocked_snapshot = cache.snapshot()
                blocked_diagnostics = cache.diagnostics()

            cache.notify_model_saved(repository, "Earthling", 1)
            refreshed_snapshot = cache.snapshot()
            refreshed_diagnostics = cache.diagnostics()

        self.assertIs(blocked_snapshot[0].model, old_model)
        self.assertEqual(blocked_diagnostics.blocked_keys, (key,))
        self.assertEqual(blocked_diagnostics.entries[key].completed_batches, 1)
        self.assertIs(blocked_diagnostics.entries[key].model, old_model)
        self.assertIsNot(refreshed_snapshot[0].model, old_model)
        self.assertEqual(refreshed_diagnostics.blocked_keys, ())
        self.assertEqual(refreshed_diagnostics.entries[key].completed_batches, 2)

    def test_initial_load_skips_active_save_and_records_blocked_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._save_model(repository, "Earthling", 1, completed_batches=1)
            coordinator = ModelSaveCoordinator()
            cache = OpponentModelCache(save_coordinator=coordinator)
            key = OpponentModelKey("Earthling", 1)

            with coordinator.saving(key):
                cache.load_initial(repository)
                blocked_snapshot = cache.snapshot()
                blocked_diagnostics = cache.diagnostics()

            cache.load_initial(repository)
            loaded_snapshot = cache.snapshot()
            loaded_diagnostics = cache.diagnostics()

        self.assertEqual(blocked_snapshot, ())
        self.assertEqual(blocked_diagnostics.blocked_keys, (key,))
        self.assertNotIn(key, blocked_diagnostics.entries)
        self.assertEqual(len(loaded_snapshot), 1)
        self.assertEqual(loaded_diagnostics.loaded_keys, (key,))

    def _save_model(
        self,
        repository: TrainingModelRepository,
        ship: str,
        slot: int,
        *,
        completed_batches: int,
    ):
        model_slot = self._create_metadata_only_slot(
            repository,
            ship,
            slot,
            completed_batches=completed_batches,
        )
        model = build_value_network(ValueNetworkConfig(8, 1))
        save_training_checkpoint(model_slot.pth_path, model)
        return model_slot

    def _create_metadata_only_slot(
        self,
        repository: TrainingModelRepository,
        ship: str,
        slot: int,
        *,
        completed_batches: int,
    ):
        metadata = metadata_from_state(
            ship=ship,
            slot=slot,
            description="Opponent",
            architecture=model_architecture_metadata(8, 1),
            training={"regimen": {"rounds_per_batch": 1}},
            progress={"completed_batches": completed_batches},
        )
        return repository.create_or_update_user_model(metadata)


if __name__ == "__main__":
    unittest.main()
