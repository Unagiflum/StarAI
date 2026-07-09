import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Objects.Ships.registry import create_ship
from src.training import torch_backend
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER
from src.training.model_registry import (
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import (
    MOVEMENT_FORWARD,
    OPPONENT_MODE_EXISTING_AI,
    OpponentSpec,
    TrainingOrchestrationConfig,
    _fully_arm_training_shofixti,
    _round_terminal_state,
    controls_for_action_index,
    discover_existing_ai_opponents,
    run_training_batch,
    run_training_round,
    simple_opponent_schedule,
)
from src.training.replay import ActionSelection, TrainingReplayBuffer, save_training_checkpoint
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
)


class FixedPolicy:
    def __init__(self, action_index):
        self.action_index = action_index

    def select_action(self, observation):
        return ActionSelection(self.action_index, exploratory=False)


class TrainingScheduleTests(unittest.TestCase):
    def test_simple_mode_schedules_every_ship_for_each_repetition(self):
        schedule = simple_opponent_schedule(2)

        self.assertEqual(len(schedule), 2 * 25)
        self.assertEqual(
            [opponent.ship for opponent in schedule[:25]],
            list(SHIP_TYPE_CATALOG_ORDER),
        )
        self.assertEqual(
            [opponent.ship for opponent in schedule[25:]],
            list(SHIP_TYPE_CATALOG_ORDER),
        )

    def test_training_action_maps_to_normal_control_aliases(self):
        controls = controls_for_action_index(15)

        self.assertEqual(
            controls,
            {
                "forward": True,
                "left": True,
                "right": False,
                "action1": False,
                "action2": True,
            },
        )


class TrainingRoundTests(unittest.TestCase):
    def test_terminal_timeout_flushes_every_pending_sample(self):
        replay = TrainingReplayBuffer(capacity=16)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            prediction_window=10,
            match_time_limit=3,
            movement_behaviors=frozenset({MOVEMENT_FORWARD}),
        )

        result = run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=FixedPolicy(0),
            replay_buffer=replay,
            config=config,
            rng=random.Random(1),
        )

        self.assertEqual(result.terminal_reason, "timeout")
        self.assertEqual(result.frames, 3)
        self.assertEqual(len(replay), 3)
        self.assertTrue(all(sample.return_value == 0.0 for sample in replay))

    def test_shofixti_is_fully_armed_for_training(self):
        ship = create_ship("Shofixti", 1)
        ship.initialize_in_battle((500, 500), 0)
        self.assertEqual(ship.shofixti_arming_stage, ship.SAFE)

        _fully_arm_training_shofixti(ship)

        self.assertEqual(ship.shofixti_arming_stage, ship.ARMED)

    def test_pending_rebirth_prevents_terminal_resolution(self):
        simulation = SimpleNamespace(
            player1=SimpleNamespace(currently_alive=False, current_hp=0),
            player2=SimpleNamespace(currently_alive=True, current_hp=10),
            aftermath=SimpleNamespace(pending_rebirths={object(): object()}),
        )

        terminal, reason = _round_terminal_state(
            simulation,
            elapsed_frames=999,
            frame_limit=1,
        )

        self.assertFalse(terminal)
        self.assertEqual(reason, "pending_rebirth")

    def test_short_deterministic_round_completes_without_rendering(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            prediction_window=2,
            match_time_limit=2,
        )

        result = run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=FixedPolicy(0),
            replay_buffer=replay,
            config=config,
            rng=random.Random(4),
        )

        self.assertEqual(result.frames, 2)
        self.assertEqual(result.terminal_reason, "timeout")
        self.assertEqual(len(replay), 2)


class ExistingAIOpponentTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_existing_ai_discovery_skips_empty_slots_and_loads_available_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=2,
                description="Opponent",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            model = build_value_network(ValueNetworkConfig(8, 1))
            save_training_checkpoint(slot.pth_path, model)

            result = discover_existing_ai_opponents(repository)

        self.assertEqual(len(result.opponents), 1)
        self.assertEqual(result.opponents[0].ship, "Earthling")
        self.assertEqual(result.opponents[0].slot, 2)
        self.assertEqual(result.skipped, ())

    def test_existing_ai_batch_uses_available_opponent_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Opponent",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            opponent_model = build_value_network(ValueNetworkConfig(8, 1))
            save_training_checkpoint(slot.pth_path, opponent_model)

            model = build_value_network(ValueNetworkConfig(8, 1))
            optimizer = build_optimizer(model, learning_rate=0.001)
            replay = TrainingReplayBuffer(capacity=16)
            config = TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                opponent_mode=OPPONENT_MODE_EXISTING_AI,
                rounds_per_batch=2,
                prediction_window=1,
                match_time_limit=1,
                minibatch_size=1,
            )

            result = run_training_batch(
                model=model,
                optimizer=optimizer,
                replay_buffer=replay,
                config=config,
                rng=random.Random(2),
                model_repository=repository,
            )

        self.assertEqual(result.completed_rounds, 2)
        self.assertEqual(len(result.round_results), 2)
        self.assertEqual(len(replay), 2)


if __name__ == "__main__":
    unittest.main()
