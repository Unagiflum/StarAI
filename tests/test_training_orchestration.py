import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Objects.Ships.registry import create_ship
from src.training import torch_backend
from src.training.contracts import OBSERVATION_INPUT_SIZE, SHIP_TYPE_CATALOG_ORDER
from src.training.model_registry import (
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import (
    OPPONENT_MODE_EXISTING_AI,
    OPPONENT_MODE_SIMPLE,
    OpponentSpec,
    SimpleOpponentController,
    TrainingBatchAborted,
    TrainingOrchestrationConfig,
    ValueNetworkPolicy,
    _fully_arm_training_shofixti,
    _opponent_controls,
    _randomize_training_start_hp,
    _round_terminal_state,
    build_deterministic_opponent_plan,
    controls_for_action_index,
    deterministic_opponent_plan_schedule,
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
    predict_action_values_read_only,
)


class FixedPolicy:
    def __init__(self, action_index):
        self.action_index = action_index
        self.selection_count = 0

    def select_action(self, observation):
        self.selection_count += 1
        return ActionSelection(self.action_index, exploratory=False)


class PendingRebirthSimulation:
    def __init__(
        self,
        _screen,
        player1,
        player2,
        *,
        audio_service=None,
        rng=None,
        include_stars=False,
        training_event_ledger=None,
    ):
        self.player1 = player1
        self.player2 = player2
        self.frame_id = 0
        self.world = []
        self.aftermath = SimpleNamespace(pending_rebirths={object(): object()})
        for player, position in (
            (self.player1, [4000.0, 4000.0]),
            (self.player2, [4100.0, 4000.0]),
        ):
            player.position = position
            player.velocity = [0.0, 0.0]
            player.rotation = 0.0
            player.current_hp = max(1, getattr(player, "current_hp", 1))
            player.currently_alive = True

    def step(self, actions=None):
        self.frame_id += 1
        return {"frame_id": self.frame_id}


class RespawningOutcomeSimulation:
    def __init__(
        self,
        _screen,
        player1,
        player2,
        *,
        audio_service=None,
        rng=None,
        include_stars=False,
        training_event_ledger=None,
    ):
        self.player1 = player1
        self.player2 = player2
        self.frame_id = 0
        self.world = []
        self.aftermath = None
        self.training_episode_deaths = ()
        self.training_spawn_initialized = True
        for player, position in (
            (self.player1, [4000.0, 4000.0]),
            (self.player2, [4100.0, 4000.0]),
        ):
            player.position = position
            player.velocity = [0.0, 0.0]
            player.rotation = 0.0
            player.current_hp = max(1, getattr(player, "current_hp", 1))
            player.currently_alive = True

    def step(self, actions=None):
        self.frame_id += 1
        self.training_episode_deaths = (2,) if self.frame_id == 1 else ()
        return {"frame_id": self.frame_id}


class SequenceRng:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        if not self.values:
            return 1.0
        return self.values.pop(0)


class SpanRng:
    def __init__(self, random_values=(), randrange_values=()):
        self.random_values = list(random_values)
        self.randrange_values = list(randrange_values)
        self.randrange_calls = 0

    def random(self):
        if not self.random_values:
            return 1.0
        return self.random_values.pop(0)

    def randrange(self, limit):
        self.randrange_calls += 1
        if not self.randrange_values:
            return 0
        value = self.randrange_values.pop(0)
        if not 0 <= value < limit:
            raise AssertionError(f"randrange value {value} outside limit {limit}")
        return value


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

class DeterministicOpponentPlanTests(unittest.TestCase):
    @staticmethod
    def trained(slot, *, ship="Earthling"):
        return OpponentSpec(
            ship=ship,
            mode=OPPONENT_MODE_EXISTING_AI,
            slot=slot,
            model=object(),
        )

    def test_plan_allocates_simple_then_slots_in_numeric_order(self):
        plan = build_deterministic_opponent_plan(
            "Earthling",
            1200,
            (self.trained(3), self.trained(1)),
            ai_opponent_chance=50.0,
            batch_number=0,
        )

        self.assertEqual(
            [
                (segment.opponent.mode, segment.opponent.slot, segment.frame_count)
                for segment in plan.segments
            ],
            [
                (OPPONENT_MODE_SIMPLE, None, 600),
                (OPPONENT_MODE_EXISTING_AI, 1, 300),
                (OPPONENT_MODE_EXISTING_AI, 3, 300),
            ],
        )
        self.assertEqual(plan.opponent_for_frame(0).mode, OPPONENT_MODE_SIMPLE)
        self.assertEqual(plan.opponent_for_frame(599).mode, OPPONENT_MODE_SIMPLE)
        self.assertEqual(plan.opponent_for_frame(600).slot, 1)
        self.assertEqual(plan.opponent_for_frame(899).slot, 1)
        self.assertEqual(plan.opponent_for_frame(900).slot, 3)
        self.assertEqual(plan.opponent_for_frame(1199).slot, 3)

    def test_plan_rotates_all_nonempty_controller_blocks_by_batch(self):
        options = (self.trained(1), self.trained(2))

        orders = []
        for batch_number in range(4):
            plan = build_deterministic_opponent_plan(
                "Earthling",
                1200,
                options,
                ai_opponent_chance=50.0,
                batch_number=batch_number,
            )
            orders.append(
                tuple(
                    "simple" if segment.opponent.slot is None
                    else f"slot-{segment.opponent.slot}"
                    for segment in plan.segments
                )
            )

        self.assertEqual(
            orders,
            [
                ("simple", "slot-1", "slot-2"),
                ("slot-1", "slot-2", "simple"),
                ("slot-2", "simple", "slot-1"),
                ("simple", "slot-1", "slot-2"),
            ],
        )

    def test_plan_uses_half_up_rounding_and_preserves_allocations_when_rotated(self):
        options = (self.trained(1), self.trained(2))
        first = build_deterministic_opponent_plan(
            "Earthling",
            11,
            options,
            ai_opponent_chance=50.0,
            batch_number=0,
        )
        rotated = build_deterministic_opponent_plan(
            "Earthling",
            11,
            options,
            ai_opponent_chance=50.0,
            batch_number=1,
        )

        self.assertEqual(
            [
                (segment.opponent.slot, segment.frame_count)
                for segment in first.segments
            ],
            [(None, 6), (1, 3), (2, 2)],
        )
        self.assertEqual(
            [
                (segment.opponent.slot, segment.frame_count)
                for segment in rotated.segments
            ],
            [(1, 3), (2, 2), (None, 6)],
        )

    def test_plan_falls_back_to_simple_when_no_trained_model_is_available(self):
        plan = build_deterministic_opponent_plan(
            "Earthling",
            17,
            (),
            ai_opponent_chance=100.0,
            batch_number=99,
        )

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].opponent.mode, OPPONENT_MODE_SIMPLE)
        self.assertEqual(plan.segments[0].frame_count, 17)

    def test_zero_and_full_ai_percent_omit_zero_length_controller_blocks(self):
        option = self.trained(2)
        simple = build_deterministic_opponent_plan(
            "Earthling",
            12,
            (option,),
            ai_opponent_chance=0.0,
        )
        trained = build_deterministic_opponent_plan(
            "Earthling",
            12,
            (option,),
            ai_opponent_chance=100.0,
        )

        self.assertEqual(
            [segment.opponent.mode for segment in simple.segments],
            [OPPONENT_MODE_SIMPLE],
        )
        self.assertEqual(
            [segment.opponent.slot for segment in trained.segments],
            [2],
        )

    def test_batch_schedule_reuses_one_plan_per_ship_for_every_round(self):
        plans = deterministic_opponent_plan_schedule(
            2,
            1200,
            (self.trained(1),),
            ai_opponent_chance=50.0,
            batch_number=1,
        )

        self.assertEqual(len(plans), 2 * len(SHIP_TYPE_CATALOG_ORDER))
        first_earthling = plans[SHIP_TYPE_CATALOG_ORDER.index("Earthling")]
        second_earthling = plans[
            len(SHIP_TYPE_CATALOG_ORDER)
            + SHIP_TYPE_CATALOG_ORDER.index("Earthling")
        ]
        self.assertIs(first_earthling, second_earthling)
        self.assertEqual(first_earthling.segments[0].opponent.slot, 1)

    def test_plan_without_models_preserves_controller_identity_and_boundaries(self):
        plan = build_deterministic_opponent_plan(
            "Earthling",
            12,
            (self.trained(1),),
            ai_opponent_chance=50.0,
        )

        stripped = plan.without_models()

        self.assertEqual(
            [
                (
                    segment.opponent.mode,
                    segment.opponent.slot,
                    segment.start_frame,
                    segment.end_frame,
                )
                for segment in stripped.segments
            ],
            [
                (
                    segment.opponent.mode,
                    segment.opponent.slot,
                    segment.start_frame,
                    segment.end_frame,
                )
                for segment in plan.segments
            ],
        )
        self.assertTrue(
            all(segment.opponent.model is None for segment in stripped.segments)
        )

    def test_frame_lookup_rejects_frames_outside_the_match(self):
        plan = build_deterministic_opponent_plan(
            "Earthling",
            12,
            (),
        )

        with self.assertRaises(IndexError):
            plan.opponent_for_frame(-1)
        with self.assertRaises(IndexError):
            plan.opponent_for_frame(12)

    def test_round_switches_controller_only_at_scheduled_match_frames(self):
        plan = build_deterministic_opponent_plan(
            "Earthling",
            6,
            (self.trained(1),),
            ai_opponent_chance=50.0,
            batch_number=0,
        )
        seen = []

        def controls(opponent, simulation, *_args):
            seen.append(
                (
                    simulation.frame_id,
                    opponent.mode,
                    opponent.slot,
                )
            )
            return {}

        with mock.patch(
            "src.training.orchestration._opponent_direct_controls",
            side_effect=controls,
        ):
            run_training_round(
                opponent=plan.initial_opponent,
                opponent_plan=plan,
                trainee_policy=FixedPolicy(0),
                replay_buffer=TrainingReplayBuffer(16),
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    gamma=0.0,
                    match_time_limit=6,
                ),
                rng=random.Random(10),
                simulation_factory=RespawningOutcomeSimulation,
                battle_view_enabled=lambda: False,
            )

        self.assertEqual(
            seen,
            [
                (0, OPPONENT_MODE_SIMPLE, None),
                (1, OPPONENT_MODE_SIMPLE, None),
                (2, OPPONENT_MODE_SIMPLE, None),
                (3, OPPONENT_MODE_EXISTING_AI, 1),
                (4, OPPONENT_MODE_EXISTING_AI, 1),
                (5, OPPONENT_MODE_EXISTING_AI, 1),
            ],
        )


class ValueNetworkPolicyTests(unittest.TestCase):
    def test_exploratory_action_is_held_for_configured_frame_span(self):
        rng = SpanRng(random_values=[0.0, 0.0], randrange_values=[7, 11])
        policy = ValueNetworkPolicy(None, epsilon=1.0, epsilon_frame_span=3, rng=rng)

        selections = [policy.select_action(()) for _ in range(4)]

        self.assertEqual([selection.action_index for selection in selections], [7, 7, 7, 11])
        self.assertTrue(all(selection.exploratory for selection in selections))
        self.assertEqual(rng.randrange_calls, 2)

    def test_reset_exploration_span_forces_next_call_to_roll_again(self):
        rng = SpanRng(random_values=[0.0, 0.0], randrange_values=[3, 9])
        policy = ValueNetworkPolicy(None, epsilon=1.0, epsilon_frame_span=8, rng=rng)

        first = policy.select_action(())
        policy.reset_exploration_span()
        second = policy.select_action(())

        self.assertEqual(first.action_index, 3)
        self.assertEqual(second.action_index, 9)
        self.assertEqual(rng.randrange_calls, 2)

    def test_greedy_span_recomputes_greedy_action_each_frame_without_extra_rolls(self):
        rng = SpanRng(random_values=[1.0])
        policy = ValueNetworkPolicy(object(), epsilon=0.5, epsilon_frame_span=3, rng=rng)
        selections = iter(
            (
                ActionSelection(1, exploratory=False),
                ActionSelection(2, exploratory=False),
                ActionSelection(3, exploratory=False),
            )
        )

        with mock.patch(
            "src.training.orchestration.select_action_epsilon_greedy",
            side_effect=lambda *args, **kwargs: next(selections),
        ) as select_action:
            actions = [policy.select_action(()) for _ in range(3)]

        self.assertEqual([selection.action_index for selection in actions], [1, 2, 3])
        self.assertFalse(any(selection.exploratory for selection in actions))
        self.assertEqual(select_action.call_count, 3)
        self.assertEqual(rng.random_values, [])


class TrainingRoundTests(unittest.TestCase):
    def test_respawned_death_and_trailing_timeout_are_separate_episodes(self):
        replay = TrainingReplayBuffer(capacity=16)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=2,
            gamma=0.0,
        )

        result = run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=FixedPolicy(0),
            replay_buffer=replay,
            config=config,
            rng=random.Random(1),
            simulation_factory=RespawningOutcomeSimulation,
            battle_view_enabled=lambda: False,
        )

        self.assertEqual(
            [episode.terminal_reason for episode in result.episode_results],
            ["resolved", "timeout"],
        )
        self.assertEqual(
            [episode.frames for episode in result.episode_results],
            [1, 1],
        )
        self.assertEqual(
            [
                (episode.win, episode.loss, episode.draw)
                for episode in result.episode_results
            ],
            [(True, False, False), (False, False, True)],
        )

    def test_terminal_timeout_flushes_every_pending_sample(self):
        replay = TrainingReplayBuffer(capacity=16)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=3,
            forward_activity=100.0,
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

    def test_nonterminal_timeout_flush_does_not_select_unapplied_action(self):
        replay = TrainingReplayBuffer(capacity=16)
        policy = FixedPolicy(0)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=2,
        )

        result = run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=policy,
            replay_buffer=replay,
            config=config,
            rng=random.Random(1),
            simulation_factory=PendingRebirthSimulation,
            battle_view_enabled=lambda: False,
        )

        self.assertEqual(result.terminal_reason, "timeout")
        self.assertEqual(result.frames, 2)
        self.assertEqual(policy.selection_count, 2)
        self.assertEqual(len(replay), 2)

    def test_simple_opponent_activity_toggles_keys_per_frame(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            forward_activity=100.0,
            a1_activity=50.0,
            a2_activity=0.0,
        )
        controller = SimpleOpponentController(config, rng=SequenceRng([0.0, 1.0]))
        simulation = SimpleNamespace(
            frame_id=0,
            player1=SimpleNamespace(position=(4000, 3900), rotation=0.0),
            player2=SimpleNamespace(position=(4000, 4000), rotation=0.0),
        )

        first = controller.controls_for_frame(simulation)
        simulation.frame_id = 1
        second = controller.controls_for_frame(simulation)

        self.assertTrue(first["forward"])
        self.assertFalse(second["forward"])
        self.assertTrue(first["action1"])
        self.assertTrue(second["action1"])
        self.assertFalse(first["action2"])
        self.assertFalse(second["action2"])

    def test_simple_opponent_face_activity_resamples_every_fps_frames(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            face_opponent_activity=50.0,
        )
        controller = SimpleOpponentController(config, rng=SequenceRng([0.0, 1.0]))
        simulation = SimpleNamespace(
            frame_id=0,
            player1=SimpleNamespace(position=(4100, 4000), rotation=0.0),
            player2=SimpleNamespace(position=(4000, 4000), rotation=0.0),
        )

        first = controller.controls_for_frame(simulation)
        simulation.frame_id = const.FPS - 1
        before_resample = controller.controls_for_frame(simulation)
        simulation.frame_id = const.FPS
        after_resample = controller.controls_for_frame(simulation)

        self.assertTrue(first["right"])
        self.assertTrue(before_resample["right"])
        self.assertFalse(after_resample["left"])
        self.assertFalse(after_resample["right"])

    def test_shofixti_is_fully_armed_for_training(self):
        ship = create_ship("Shofixti", 1)
        ship.initialize_in_battle((500, 500), 0)
        self.assertEqual(ship.shofixti_arming_stage, ship.SAFE)

        _fully_arm_training_shofixti(ship)

        self.assertEqual(ship.shofixti_arming_stage, ship.ARMED)

    def test_training_start_hp_uses_linear_random_ceiled_with_minimum(self):
        ship = SimpleNamespace(start_hp=10, current_hp=10)

        _randomize_training_start_hp(ship, SequenceRng([0.0]))
        self.assertEqual(ship.current_hp, 1)

        _randomize_training_start_hp(ship, SequenceRng([0.25]))
        self.assertEqual(ship.current_hp, 3)

        _randomize_training_start_hp(ship, SequenceRng([0.99]))
        self.assertEqual(ship.current_hp, 10)

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

    def test_seeded_causal_round_targets_are_deterministic(self):
        signatures = []
        for _ in range(2):
            replay = TrainingReplayBuffer(capacity=16)
            result = run_training_round(
                opponent=OpponentSpec("Earthling"),
                trainee_policy=FixedPolicy(6),
                replay_buffer=replay,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    match_time_limit=8,
                ),
                rng=random.Random(731),
            )
            signatures.append(
                (
                    result.frames,
                    result.terminal_reason,
                    tuple(result.component_totals.items()),
                    tuple(
                        (
                            replay[index].observation,
                            replay[index].action_index,
                            replay[index].return_value,
                        )
                        for index in range(len(replay))
                    ),
                )
            )

        self.assertEqual(signatures[0], signatures[1])

    def test_display_toggle_does_not_change_headless_training_semantics(self):
        results = []
        replay_lengths = []
        for display_on in (False, True):
            replay = TrainingReplayBuffer(capacity=8)
            config = TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                match_time_limit=2,
                display_on=display_on,
            )

            results.append(
                run_training_round(
                    opponent=OpponentSpec("Earthling"),
                    trainee_policy=FixedPolicy(0),
                    replay_buffer=replay,
                    config=config,
                    rng=random.Random(8),
                )
            )
            replay_lengths.append(len(replay))

        self.assertEqual(results[0].frames, results[1].frames)
        self.assertEqual(results[0].terminal_reason, results[1].terminal_reason)
        self.assertEqual(results[0].total_return, results[1].total_return)
        self.assertEqual(replay_lengths, [2, 2])

    def test_training_round_publishes_battle_view_progress(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            gamma=0.0,
            match_time_limit=1,
        )
        events = []

        run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=FixedPolicy(0),
            replay_buffer=replay,
            config=config,
            rng=random.Random(9),
            progress_callback=events.append,
        )

        views = [event["battle_view"] for event in events if "battle_view" in event]
        self.assertTrue(views)
        self.assertIn("game_objects", views[0])
        self.assertIn("border_rect", views[0])
        self.assertEqual(views[-1]["frame_id"], 1)

    def test_training_round_skips_battle_view_when_disabled(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            gamma=0.0,
            match_time_limit=1,
        )
        events = []

        with mock.patch(
            "src.training.orchestration._battle_view_from_simulation"
        ) as build_view:
            run_training_round(
                opponent=OpponentSpec("Earthling"),
                trainee_policy=FixedPolicy(0),
                replay_buffer=replay,
                config=config,
                rng=random.Random(9),
                progress_callback=events.append,
                battle_view_enabled=lambda: False,
            )

        build_view.assert_not_called()
        self.assertFalse(any("battle_view" in event for event in events))
        self.assertTrue(any(event.get("event") == "frame" for event in events))

    def test_disabled_battle_view_avoids_display_throttle(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            display_on=True,
            gamma=0.0,
            match_time_limit=1,
        )

        with (
            mock.patch(
                "src.training.orchestration.time.perf_counter",
                return_value=100.0,
            ),
            mock.patch("src.training.orchestration.time.sleep") as sleep,
        ):
            run_training_round(
                opponent=OpponentSpec("Earthling"),
                trainee_policy=FixedPolicy(0),
                replay_buffer=replay,
                config=config,
                rng=random.Random(9),
                battle_view_enabled=lambda: False,
            )

        sleep.assert_not_called()

    def test_training_batch_announces_optimization_phase(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

        model = build_value_network(ValueNetworkConfig(8, 1))
        optimizer = build_optimizer(model, learning_rate=0.001)
        replay = TrainingReplayBuffer(capacity=16)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            rounds_per_batch=1,
            gamma=0.0,
            match_time_limit=1,
            minibatch_size=1,
            replay_updates_per_batch=2,
            hidden_layer_width=8,
            hidden_layer_count=1,
        )
        events = []

        run_training_batch(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay,
            config=config,
            rng=random.Random(12),
            progress_callback=events.append,
        )

        optimization_events = [
            event for event in events if event.get("event") == "batch_optimization_start"
        ]
        self.assertEqual(len(optimization_events), 1)
        self.assertEqual(optimization_events[0]["replay_updates"], 2)
        self.assertEqual(optimization_events[0]["replay_size"], len(replay))

    def test_training_batch_finishes_optimization_after_stop_is_requested(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            rounds_per_batch=1,
            replay_updates_per_batch=3,
        )
        stop_requested = [False]
        optimize_calls = []

        def optimize(*args, **kwargs):
            optimize_calls.append((args, kwargs))
            if len(optimize_calls) == 1:
                stop_requested[0] = True
            return SimpleNamespace(loss=float(len(optimize_calls)))

        with (
            mock.patch(
                "src.training.orchestration.run_training_round",
                return_value=SimpleNamespace(),
            ),
            mock.patch(
                "src.training.orchestration.optimize_from_replay",
                side_effect=optimize,
            ),
        ):
            result = run_training_batch(
                model=object(),
                optimizer=object(),
                replay_buffer=replay,
                config=config,
                stop_requested=lambda: stop_requested[0],
            )

        self.assertTrue(stop_requested[0])
        self.assertEqual(len(optimize_calls), 3)
        self.assertEqual(result.optimization_losses, (1.0, 2.0, 3.0))

    def test_training_round_aborts_when_stop_is_requested_mid_round(self):
        replay = TrainingReplayBuffer(capacity=8)
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            gamma=0.0,
            match_time_limit=10,
        )
        stop_requested = [False]

        def on_progress(payload):
            if payload.get("event") == "frame":
                stop_requested[0] = True

        with self.assertRaises(TrainingBatchAborted):
            run_training_round(
                opponent=OpponentSpec("Earthling"),
                trainee_policy=FixedPolicy(0),
                replay_buffer=replay,
                config=config,
                rng=random.Random(10),
                progress_callback=on_progress,
                stop_requested=lambda: stop_requested[0],
            )

        self.assertEqual(len(replay), 1)


class ExistingAIOpponentTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_trained_opponent_controls_use_read_only_inference(self):
        simulation = SimpleNamespace(
            player1=SimpleNamespace(),
            player2=SimpleNamespace(),
            frame_id=12,
            world=(),
        )
        opponent = OpponentSpec("Earthling", model=object())
        config = TrainingOrchestrationConfig(trainee_ship="Earthling")
        simple_controller = mock.Mock()

        with (
            mock.patch(
                "src.training.orchestration.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.orchestration.select_action_epsilon_greedy",
                return_value=ActionSelection(0, exploratory=False),
            ) as select_action,
        ):
            controls = _opponent_controls(
                opponent,
                simulation,
                config,
                simple_controller,
            )

        self.assertEqual(controls, controls_for_action_index(0))
        self.assertIs(
            select_action.call_args.kwargs["value_predictor"],
            predict_action_values_read_only,
        )
        simple_controller.controls_for_frame.assert_not_called()

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

    def test_existing_ai_batch_uses_mixed_all_ship_schedule(self):
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
            replay = TrainingReplayBuffer(capacity=64)
            config = TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                opponent_mode=OPPONENT_MODE_EXISTING_AI,
                rounds_per_batch=2,
                gamma=0.0,
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

        self.assertEqual(result.completed_rounds, 2 * len(SHIP_TYPE_CATALOG_ORDER))
        self.assertEqual(len(result.round_results), 2 * len(SHIP_TYPE_CATALOG_ORDER))
        self.assertEqual(len(replay), 2 * len(SHIP_TYPE_CATALOG_ORDER))


if __name__ == "__main__":
    unittest.main()
