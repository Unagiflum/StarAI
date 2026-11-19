import unittest
from types import SimpleNamespace
from unittest import mock

import src.const as const
from src.Battle.battle_ai import BattleAIManager
from src.Battle.computer_control import (
    computer_action2_allowed,
    guard_computer_controls,
)
from src.training.action_safety import guard_training_selection
from src.training.baseline import BaselineController, BaselineOpponentController
from src.training.contracts import (
    ACTION_INDEX_TABLE,
    OBSERVATION_INPUT_SIZE,
    TrainingAction,
)
from src.training.replay_contracts import ActionSelection


def ship(name, *, position=(1000, 1000), **attributes):
    values = {
        "name": name,
        "position": list(position),
        "currently_alive": True,
        "current_hp": 10,
    }
    values.update(attributes)
    return SimpleNamespace(**values)


class ComputerActionSafetyTests(unittest.TestCase):
    def test_druuge_a2_requires_less_energy_than_one_primary_shot(self):
        druuge = ship("Druuge", current_energy=3, a1_cost=4)
        enemy = ship("Earthling")

        self.assertTrue(computer_action2_allowed(druuge, enemy))
        druuge.current_energy = 4
        self.assertFalse(computer_action2_allowed(druuge, enemy))
        druuge.current_energy = 5
        self.assertFalse(computer_action2_allowed(druuge, enemy))

    def test_shofixti_counts_cloaked_living_enemy_in_range(self):
        shofixti = ship("Shofixti", position=(1000, 1000))
        enemy = ship(
            "Ilwrath",
            position=(1000, 1700),
            cloaked=True,
            trackable=False,
        )

        self.assertTrue(computer_action2_allowed(shofixti, enemy))

    def test_shofixti_blocks_dead_or_out_of_range_enemy(self):
        shofixti = ship("Shofixti", position=(1000, 1000))
        enemy = ship("Earthling", position=(1000, 1721))

        self.assertFalse(computer_action2_allowed(shofixti, enemy))
        enemy.position = [1000, 1100]
        enemy.current_hp = 0
        self.assertFalse(computer_action2_allowed(shofixti, enemy))

    def test_shofixti_range_uses_toroidal_distance(self):
        shofixti = ship("Shofixti", position=(const.ARENA_SIZE - 10, 1000))
        enemy = ship("Earthling", position=(10, 1000))

        self.assertTrue(computer_action2_allowed(shofixti, enemy))

    def test_guard_preserves_other_controls_while_clearing_a2(self):
        druuge = ship("Druuge", current_energy=4, a1_cost=4)
        enemy = ship("Earthling")
        controls = TrainingAction.from_mask(29)

        guarded = guard_computer_controls(controls, druuge, enemy)

        self.assertEqual(guarded.mask, 13)
        self.assertTrue(guarded.thrust)
        self.assertTrue(guarded.turn_right)
        self.assertTrue(guarded.a1)
        self.assertFalse(guarded.a2)

    def test_training_selection_records_the_executed_non_a2_action(self):
        druuge = ship("Druuge", current_energy=4, a1_cost=4)
        selection = ActionSelection(23, exploratory=True, action_values=(1.0,))

        guarded = guard_training_selection(selection, druuge, ship("Earthling"))

        self.assertEqual(guarded.action_index, 11)
        self.assertTrue(guarded.exploratory)
        self.assertEqual(guarded.action_values, (1.0,))


class ComputerShipInitializationTests(unittest.TestCase):
    def test_battle_ai_arms_only_computer_controlled_shofixti(self):
        human = ship("Shofixti", shofixti_arming_stage=0, ARMED=2)
        computer = ship("Shofixti", shofixti_arming_stage=0, ARMED=2)
        simulation = SimpleNamespace(player1=human, player2=computer)
        manager = BattleAIManager({1: False, 2: True})

        with mock.patch.object(manager, "_resolve_model", return_value=(None, [])):
            manager.bind_round(simulation)

        self.assertEqual(human.shofixti_arming_stage, 0)
        self.assertEqual(computer.shofixti_arming_stage, 2)

    def test_battle_ai_guards_fallback_or_model_controls(self):
        druuge = ship("Druuge", current_energy=4, a1_cost=4)
        enemy = ship("Earthling")
        simulation = SimpleNamespace(player1=druuge, player2=enemy)
        controller = mock.Mock()
        controller.actions_for_frame.return_value = {
            "forward": True,
            "left": False,
            "right": False,
            "action1": True,
            "action2": True,
        }
        manager = BattleAIManager({1: True})
        manager._controllers[1] = controller

        actions = manager.actions_for_frame(simulation)

        self.assertTrue(actions[1]["action1"])
        self.assertFalse(actions[1]["action2"])

    def test_baseline_opponent_controller_applies_druuge_guard(self):
        simulation = SimpleNamespace(
            frame_id=0,
            world=(),
            player1=ship("Earthling"),
            player2=ship("Druuge", current_energy=4, a1_cost=4),
        )
        a2_only = next(
            index for index, action in enumerate(ACTION_INDEX_TABLE) if action.mask == 16
        )
        with (
            mock.patch(
                "src.training.observation.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch.object(BaselineController, "decide", return_value=a2_only),
        ):
            controller = BaselineOpponentController(action_frame_span=1)
            blocked = controller.direct_controls_for_frame(simulation)
            self.assertFalse(blocked.a2)

            simulation.player2.current_energy = 3
            allowed = controller.direct_controls_for_frame(simulation)
            self.assertTrue(allowed.a2)


if __name__ == "__main__":
    unittest.main()
