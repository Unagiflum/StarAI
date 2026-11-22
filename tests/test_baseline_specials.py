"""Specials and mode switches named in the shared behaviour table: each runs
in real fights, late specials run after the band, and whole-rule specials
fly exactly as the ship's own strategy."""

import dataclasses
import unittest
from unittest import mock

from src.training.baseline import strategies
from src.training.baseline.arena import ArenaMatch, run_match
from src.training.baseline.behaviours import Band, ShipBehaviour, ship_behaviour

# Every special and mode switch, with the ship that uses it, an opponent
# that exercises it, and whether it runs late.
SPECIALS = {
    "Androsynth": ("Chmmr", ("androsynth_chmmr",), (), "androsynth_blazer"),
    "Arilou": ("Chmmr", ("arilou_chmmr",), (), None),
    "Chenjesu": ("Mycon", ("chenjesu_dogi", "chenjesu_crystal"), (), None),
    "Chmmr": ("Mycon", ("chmmr_tractor",), (), None),
    "Druuge": ("Chenjesu", ("druuge_recoil_escape", "druuge_dogi_targets", "druuge_crew_battery"), (), None),
    "Earthling": ("Mycon", ("earthling_point_defense", "earthling_nuke", "earthling_whip"), (), None),
    "Ilwrath": ("Mycon", ("ilwrath_flame_cloak", "ilwrath_hang_back", "ilwrath_follow"), (), None),
    "KohrAh": ("Spathi", ("kohr_ah_gas", "kohr_ah_saw"), (), None),
    "KzerZa": ("Mycon", ("kzer_za_fighters",), (), None),
    "Mmrnmrhm": ("Mycon", ("mmrnmrhm_attack",), ("face_opponent",), "mmrnmrhm_forms"),
    "Melnorme": ("Mycon", ("melnorme_charge",), ("melnorme_aim",), None),
    "Mycon": ("Spathi", ("mycon_regenerate",), ("gravity_whip_when_slow", "mycon_plasmoid"), None),
    "Orz": ("Chmmr", ("orz_marines",), ("orz_turret",), None),
    "Pkunk": ("Mycon", ("pkunk_insult", "pkunk_pass"), ("pkunk_gun",), None),
    "Shofixti": ("Mycon", ("shofixti_glory",), (), None),
    "Slylandro": ("Chmmr", ("slylandro_harvest", "slylandro_low_harvest"), (), None),
    "Supox": ("Chmmr", ("supox_slides",), (), None),
    "Syreen": ("Chmmr", ("syreen_song", "syreen_pods", "syreen_close_in", "syreen_chmmr"), ("syreen_gun",), None),
    "Utwig": ("Chmmr", ("utwig_chmmr",), (), None),
    "Yehat": ("Shofixti", ("shield_glory", "yehat_chmmr"), (), None),
    "ZoqFotPik": ("Mycon", ("zoq_fot_pik_tongue", "zoq_fot_pik_lunge"), ("zoq_fot_pik_gun",), None),
    "Thraddash": ("Chmmr", ("thraddash_rules",), (), None),
    "Umgah": ("Spathi", ("umgah_rules",), (), None),
    "Vux": ("Chmmr", ("vux_rules",), (), None),
}


class SpecialRuleTests(unittest.TestCase):
    def test_every_special_and_mode_switch_is_covered(self):
        specials = {name for _, early, late, _ in SPECIALS.values() for name in early + late}
        switches = {switch for *_, switch in SPECIALS.values() if switch}
        self.assertEqual(specials, set(strategies.SPECIAL_RULES))
        self.assertEqual(switches, set(strategies.FORM_SWITCHES))

    def test_every_special_flies_in_a_real_fight(self):
        for ship, (opponent, early, late, switch) in SPECIALS.items():
            with self.subTest(ship=ship):
                entry = dataclasses.replace(
                    ship_behaviour(ship),
                    specials=early,
                    late_specials=late,
                    form_switch=switch,
                    band=Band(500.0, 1000.0),
                    opponents={},
                )
                result = run_match(ArenaMatch(ship, opponent, 0, "specials", entry=entry), frames=90)
                self.assertEqual(result.frames, 90)

    def test_whole_rule_specials_fly_as_the_ship_strategy(self):
        # The Thraddash, Umgah and Vux run from their table entries, each a
        # whole-rule special, fight for fight as their earlier rules.
        for ship in ("Thraddash", "Umgah", "Vux"):
            self.assertIs(strategies.STRATEGIES[ship], strategies.shared)
            for opponent in ("Chmmr", "Orz"):
                with self.subTest(ship=ship, opponent=opponent):
                    earlier = run_match(ArenaMatch(ship, opponent, 1, "earlier"), frames=240)
                    moved = run_match(ArenaMatch(ship, opponent, 1), frames=240)
                    self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)


class MovedShipTests(unittest.TestCase):
    def test_the_druuge_entry_flies_as_its_earlier_rules(self):
        for opponent in ("Chenjesu", "KzerZa", "Mycon"):
            with self.subTest(opponent=opponent):
                earlier = run_match(ArenaMatch("Druuge", opponent, 2, "earlier"), frames=240)
                moved = run_match(ArenaMatch("Druuge", opponent, 2), frames=240)
                self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)


class LateSpecialTests(unittest.TestCase):
    def test_late_specials_run_after_the_band_and_before_the_weapons(self):
        seen = {}

        def record(ctx):
            seen["steering"] = ctx.plan.steering_claimed
            seen["a1"] = ctx.plan.a1

        from tests.test_baseline_strategy import _rule_context, _seen

        entry = ShipBehaviour(
            band=Band(500.0, 1000.0),
            late_specials=("record",),
            weapons=(None, None),
        )
        with mock.patch.dict(strategies.SPECIAL_RULES, {"record": record}):
            context = _rule_context("Spathi", enemy=_seen(10.0, 3000.0), enemy_name="Mycon")
            context.behaviour = entry
            strategies.shared(context)

        # The band closed in from beyond its far edge before the late rule.
        self.assertTrue(seen["steering"])
        self.assertIsNone(seen["a1"])


if __name__ == "__main__":
    unittest.main()
