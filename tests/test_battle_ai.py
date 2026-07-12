import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.Battle.battle import (
    filter_ai_key_changes,
    reset_ai_player_inputs,
)
from src.Battle.battle_ai import (
    BattleAIManager,
    FallbackController,
    controls_for_action_index,
    load_battle_ai_model,
)
from src.training.model_loader import (
    InferenceModelCache,
    LoadedInferenceModel,
    InferenceModelKey,
)
from src.training.model_registry import (
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
)


class SequenceRng:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


class StaticInferenceModelCache:
    def __init__(self, entries=(), errors=None):
        self.entries = {
            (entry.key.ship, entry.key.slot): entry
            for entry in entries
        }
        self.errors = dict(errors or {})

    def entry_for(self, ship, slot):
        return self.entries.get((str(ship), int(slot)))

    def error_for(self, ship, slot):
        return self.errors.get((str(ship), int(slot)))


def make_ship(name, player, position=(4000, 4000), rotation=0.0):
    return SimpleNamespace(
        name=name,
        player=player,
        position=list(position),
        rotation=rotation,
        currently_alive=True,
        current_hp=10,
    )


class BattleAIModelResolutionTests(unittest.TestCase):
    def test_model_loading_uses_preferred_device_for_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            slot = self._write_user_slot(repository, "Earthling", 1, "Default")
            model = mock.Mock()

            with (
                mock.patch(
                    "src.training.model_loader.torch_backend.get_torch",
                    return_value=object(),
                ),
                mock.patch(
                    "src.training.model_loader.torch_backend.preferred_device",
                    return_value="cuda",
                ),
                mock.patch(
                    "src.training.model_loader.build_value_network",
                    return_value=model,
                ) as build_network,
                mock.patch(
                    "src.training.model_loader.load_training_checkpoint"
                ) as load_checkpoint,
            ):
                loaded = load_battle_ai_model(slot)

        self.assertIs(loaded.model, model)
        build_network.assert_called_once()
        self.assertEqual(build_network.call_args.kwargs["device"], "cuda")
        load_checkpoint.assert_called_once_with(
            slot.pth_path,
            model,
            map_location="cuda",
        )
        model.eval.assert_called_once_with()

    def test_default_model_priority_precedes_first_non_default_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Aggressive")
            preferred_slot = self._write_user_slot(repository, "Earthling", 2, "Default")
            fallback_slot = repository.slot_for("Earthling", 1)
            cache = StaticInferenceModelCache(
                (
                    self._loaded_entry(preferred_slot, object()),
                    self._loaded_entry(fallback_slot, object()),
                )
            )

            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_cache=cache,
            )
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "Earthling-02")

    def test_default_load_failure_falls_back_to_first_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Default")
            backup_slot = self._write_user_slot(repository, "Earthling", 3, "Backup")
            cache = StaticInferenceModelCache(
                (self._loaded_entry(backup_slot, object()),),
                errors={("Earthling", 1): "bad checkpoint"},
            )

            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_cache=cache,
            )
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "Earthling-03")
        self.assertIn("bad checkpoint", manager.load_failures[1][0])

    def test_load_failure_results_in_fallback_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Default")
            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_cache=StaticInferenceModelCache(
                    errors={("Earthling", 1): "nope"},
                ),
            )
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "None found")
        self.assertEqual(
            set(manager.actions_for_frame(simulation)[1]),
            {"forward", "left", "right", "action1", "action2"},
        )

    def test_pytorch_unavailable_results_in_fallback_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Default")
            model_cache = InferenceModelCache()
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            with mock.patch("src.training.torch_backend._torch", None):
                model_cache.load_initial(repository)
                manager = BattleAIManager(
                    {1: True},
                    repository=repository,
                    model_cache=model_cache,
                )
                manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "None found")
        self.assertIn("PyTorch unavailable", manager.load_failures[1][0])

    def test_human_player_has_no_label(self):
        manager = BattleAIManager({1: False, 2: True})

        self.assertIsNone(manager.label_for_player(1))

    def _write_user_slot(self, repository, ship, slot, description):
        metadata = metadata_from_state(
            ship=ship,
            slot=slot,
            description=description,
            architecture=model_architecture_metadata(8, 1),
            training={"regimen": {"rounds_per_batch": 1}},
        )
        model_slot = repository.create_or_update_user_model(metadata)
        model_slot.pth_path.write_bytes(b"checkpoint")
        return model_slot

    def _loaded_entry(self, slot, model):
        return LoadedInferenceModel(
            key=InferenceModelKey(slot.ship, slot.slot),
            model=model,
            slot=slot,
            description=slot.description,
        )


class BattleAIFallbackControllerTests(unittest.TestCase):
    def test_fallback_faces_enemy_and_holds_forward(self):
        controller = FallbackController(1, rng=SequenceRng([1.0, 1.0]))
        simulation = SimpleNamespace(
            player1=make_ship("Earthling", 1, position=(4000, 4000), rotation=0.0),
            player2=make_ship("Chenjesu", 2, position=(4100, 4000), rotation=0.0),
        )

        actions = controller.actions_for_frame(simulation)

        self.assertTrue(actions["forward"])
        self.assertFalse(actions["left"])
        self.assertTrue(actions["right"])
        self.assertFalse(actions["action1"])
        self.assertFalse(actions["action2"])

    def test_fallback_button_probabilities_toggle_hold_state(self):
        controller = FallbackController(1, rng=SequenceRng([0.0, 0.0, 1.0, 1.0]))
        simulation = SimpleNamespace(
            player1=make_ship("Earthling", 1, position=(4000, 4000), rotation=0.0),
            player2=make_ship("Chenjesu", 2, position=(4000, 3900), rotation=0.0),
        )

        first = controller.actions_for_frame(simulation)
        second = controller.actions_for_frame(simulation)

        self.assertTrue(first["action1"])
        self.assertTrue(first["action2"])
        self.assertTrue(second["action1"])
        self.assertTrue(second["action2"])

    def test_training_action_index_maps_to_battle_controls(self):
        self.assertEqual(
            controls_for_action_index(15),
            {
                "forward": True,
                "left": True,
                "right": False,
                "action1": False,
                "action2": True,
            },
        )


class BattleAIInputIntegrationTests(unittest.TestCase):
    def test_filter_removes_ai_player_action_keys_only(self):
        simulation = SimpleNamespace(
            settings={
                "Player 1: Forward": 10,
                "Player 1: Left": 11,
                "Player 1: Right": 12,
                "Player 1: Action 1": 13,
                "Player 1: Action 2": 14,
                "Player 2: Forward": 20,
                "Player 2: Left": 21,
                "Player 2: Right": 22,
                "Player 2: Action 1": 23,
                "Player 2: Action 2": 24,
            },
        )

        def binding_for_key(key):
            for player in (1, 2):
                for control in ("Forward", "Left", "Right", "Action 1", "Action 2"):
                    if key == simulation.settings[f"Player {player}: {control}"]:
                        return player, control
            return None

        simulation._binding_for_key = binding_for_key
        manager = BattleAIManager({1: True, 2: False})

        filtered = filter_ai_key_changes(
            simulation,
            [(10, True), (20, True), (999, True), (13, False), (24, False)],
            manager,
        )

        self.assertEqual(filtered, [(20, True), (999, True), (24, False)])

    def test_reset_ai_player_inputs_clears_stale_ai_ship_controls(self):
        player1 = mock.Mock()
        player2 = mock.Mock()
        simulation = SimpleNamespace(
            player1=player1,
            player2=player2,
            settings={
                "Player 1: Forward": 10,
                "Player 1: Left": 11,
                "Player 1: Right": 12,
                "Player 1: Action 1": 13,
                "Player 1: Action 2": 14,
                "Player 2: Forward": 20,
                "Player 2: Left": 21,
                "Player 2: Right": 22,
                "Player 2: Action 1": 23,
                "Player 2: Action 2": 24,
            },
            key_states={
                10: True,
                11: True,
                12: True,
                13: True,
                14: True,
                20: True,
                21: True,
                22: True,
                23: True,
                24: True,
            },
        )
        manager = BattleAIManager({1: True, 2: False})

        reset_ai_player_inputs(simulation, manager)

        self.assertEqual(
            {key: simulation.key_states[key] for key in (10, 11, 12, 13, 14)},
            {10: False, 11: False, 12: False, 13: False, 14: False},
        )
        self.assertEqual(
            {key: simulation.key_states[key] for key in (20, 21, 22, 23, 24)},
            {20: True, 21: True, 22: True, 23: True, 24: True},
        )
        player1.reset_controls.assert_called_once_with()
        player2.reset_controls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
