import unittest
from dataclasses import dataclass

from src.menu_state import FleetModel, FleetShip, ShipSelectionState


@dataclass
class Ship:
    name: str
    currently_alive: bool = True
    current_hp: int = 10


class FleetModelTests(unittest.TestCase):
    def test_mutation_preserves_order_cost_and_capacity(self):
        fleet = FleetModel(capacity=3)

        self.assertTrue(fleet.add_ship("Spathi", 18))
        self.assertTrue(fleet.add_ship("Earthling", 11))
        self.assertTrue(fleet.add_ship("Spathi", 18))
        self.assertFalse(fleet.add_ship("Arilou", 16))
        self.assertEqual(
            fleet.ship_names, ("Spathi", "Earthling", "Spathi")
        )
        self.assertEqual(fleet.ship_slots, ("Spathi", "Earthling", "Spathi"))
        self.assertEqual(fleet.total_cost, 47)

        removed = fleet.remove_ship(1)
        self.assertEqual(removed, FleetShip("Earthling", 11))
        self.assertEqual(fleet.ship_names, ("Spathi", "Spathi"))
        self.assertEqual(fleet.ship_slots, ("Spathi", None, "Spathi"))
        self.assertEqual(fleet.total_cost, 36)

        replaced = fleet.replace_ship(1, "Arilou", 16)
        self.assertIsNone(replaced)
        self.assertEqual(fleet.ship_names, ("Spathi", "Arilou", "Spathi"))
        self.assertEqual(fleet.total_cost, 52)

    def test_replace_and_clear_apply_the_same_capacity_rule(self):
        fleet = FleetModel(capacity=2)
        fleet.add_ship("Old", 1)

        fleet.replace([
            FleetShip("First", 2),
            FleetShip("Second", 3),
            FleetShip("Overflow", 100),
        ])
        self.assertEqual(fleet.ship_names, ("First", "Second"))
        self.assertEqual(fleet.total_cost, 5)

        fleet.clear()
        self.assertTrue(fleet.is_empty)
        self.assertEqual(fleet.total_cost, 0)


class ShipSelectionStateTests(unittest.TestCase):
    def state(self, player1, player2, **kwargs):
        return ShipSelectionState(
            {1: player1, 2: player2},
            {
                1: [ship.name for ship in player1],
                2: [ship.name for ship in player2],
            },
            **kwargs,
        )

    def test_survivor_is_preselected_and_cannot_be_replaced(self):
        survivor = Ship("Survivor")
        reserve = Ship("Reserve")
        opponent = Ship("Opponent")
        state = self.state(
            [survivor, reserve],
            [opponent],
            preselected={1: survivor, 2: None},
        )

        self.assertEqual(state.survivor_locked_players, frozenset({1}))
        self.assertEqual(state.selection(1).ship, survivor)
        self.assertFalse(state.selection_allowed(1))
        self.assertFalse(state.select_index(1, 1))
        self.assertEqual(state.selection(1).ship, survivor)

        self.assertTrue(state.select_index(2, 0))
        self.assertTrue(state.confirmation_ready)
        self.assertEqual(state.selected_ships(), (survivor, opponent))

    def test_forced_order_uses_tie_break_loser_second(self):
        player1 = [Ship("P1-A"), Ship("P1-B")]
        player2 = [Ship("P2-A"), Ship("P2-B")]
        state = self.state(
            player1, player2, choose_second_player=1
        )

        self.assertEqual(state.first_player, 2)
        self.assertEqual(state.active_player, 2)
        self.assertFalse(state.selection_allowed(1))
        self.assertFalse(state.select_index(1, 0))

        self.assertTrue(state.select_index(2, 1))
        self.assertTrue(state.first_locked)
        self.assertEqual(state.active_player, 1)
        self.assertFalse(state.selection_allowed(2))
        self.assertFalse(state.select_index(2, 0))

        self.assertTrue(state.select_index(1, 0))
        self.assertTrue(state.confirmation_ready)
        self.assertEqual(state.selected_ships(), (player1[0], player2[1]))

        # The second chooser remains active, matching the menu's prior behavior.
        self.assertTrue(state.select_index(1, 1))
        self.assertEqual(state.selected_ships(), (player1[1], player2[1]))

    def test_dead_ships_are_excluded_from_manual_and_random_candidates(self):
        dead = Ship("Dead", currently_alive=False, current_hp=0)
        alive = Ship("Alive")
        opponent = Ship("Opponent")
        state = self.state([dead, alive], [opponent])

        self.assertEqual(state.alive_indices(1), (1,))
        self.assertFalse(state.select_index(1, 0))
        self.assertTrue(state.select_index(1, state.alive_indices(1)[0]))
        self.assertEqual(state.selection(1).ship, alive)

    def test_confirmation_tracks_selected_ship_liveness(self):
        player1 = Ship("P1")
        player2 = Ship("P2")
        state = self.state([player1], [player2])
        state.select_index(1, 0)
        state.select_index(2, 0)
        self.assertTrue(state.confirmation_ready)

        player2.currently_alive = False
        self.assertFalse(state.confirmation_ready)
        self.assertIsNone(state.selected_ships())

    def test_selected_ship_toggles_off_and_blank_click_can_deselect(self):
        player1 = Ship("P1")
        player2 = Ship("P2")
        state = self.state([player1], [player2])

        self.assertTrue(state.toggle_index(1, 0))
        self.assertEqual(state.selection(1).ship, player1)
        self.assertTrue(state.toggle_index(1, 0))
        self.assertIsNone(state.selection(1))

        state.select_index(1, 0)
        self.assertTrue(state.deselect(1))
        self.assertIsNone(state.selection(1))

    def test_locked_selection_cannot_be_toggled_or_deselected(self):
        survivor = Ship("Survivor")
        state = self.state(
            [survivor],
            [Ship("Opponent")],
            preselected={1: survivor},
        )

        self.assertFalse(state.toggle_index(1, 0))
        self.assertFalse(state.deselect(1))
        self.assertEqual(state.selection(1).ship, survivor)


if __name__ == "__main__":
    unittest.main()
