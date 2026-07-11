import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.Battle.battle_ai import (
    BattleAIManager,
    BattleAIModel,
    FallbackController,
    controls_for_action_index,
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
    def test_default_model_priority_precedes_first_non_default_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Aggressive")
            self._write_user_slot(repository, "Earthling", 2, "Default")
            loaded_slots = []

            def loader(slot):
                loaded_slots.append(slot.slot)
                return BattleAIModel(object(), slot, f"loaded-{slot.slot}")

            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_loader=loader,
            )
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            manager.bind_round(simulation)

        self.assertEqual(loaded_slots, [2])
        self.assertEqual(manager.label_for_player(1), "loaded-2")

    def test_default_load_failure_falls_back_to_first_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Default")
            self._write_user_slot(repository, "Earthling", 3, "Backup")

            def loader(slot):
                if slot.slot == 1:
                    raise RuntimeError("bad checkpoint")
                return BattleAIModel(object(), slot, f"loaded-{slot.slot}")

            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_loader=loader,
            )
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "loaded-3")
        self.assertIn("bad checkpoint", manager.load_failures[1][0])

    def test_load_failure_results_in_fallback_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            self._write_user_slot(repository, "Earthling", 1, "Default")
            manager = BattleAIManager(
                {1: True},
                repository=repository,
                model_loader=lambda slot: (_ for _ in ()).throw(RuntimeError("nope")),
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
            manager = BattleAIManager({1: True}, repository=repository)
            simulation = SimpleNamespace(
                player1=make_ship("Earthling", 1),
                player2=make_ship("Chenjesu", 2),
            )

            with mock.patch("src.training.torch_backend._torch", None):
                manager.bind_round(simulation)

        self.assertEqual(manager.label_for_player(1), "None found")

    def test_human_player_has_no_label(self):
        manager = BattleAIManager({1: False, 2: True}, model_loader=lambda slot: object())

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


if __name__ == "__main__":
    unittest.main()
