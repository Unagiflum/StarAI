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

    def test_the_slylandro_entry_flies_its_earlier_chmmr_rules(self):
        # Against the Chmmr the entry's override (low harvest, a 700-850
        # orbit band, lightning within 1,000) is the earlier rule.
        self.assertIs(strategies.STRATEGIES["Slylandro"], strategies.shared)
        for cell in (("strong", "strong"), ("weak", "strong")):
            with self.subTest(cell=cell):
                earlier = run_match(ArenaMatch("Slylandro", "Chmmr", 1, "earlier", *cell), frames=240)
                moved = run_match(ArenaMatch("Slylandro", "Chmmr", 1, "default", *cell), frames=240)
                self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_syreen_entry_flies_its_earlier_chmmr_rule(self):
        self.assertIs(strategies.STRATEGIES["Syreen"], strategies.shared)
        earlier = run_match(ArenaMatch("Syreen", "Chmmr", 1, "earlier"), frames=240)
        moved = run_match(ArenaMatch("Syreen", "Chmmr", 1), frames=240)
        self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_syreen_recharges_below_the_songs_cost(self):
        behaviour = ship_behaviour("Syreen", "Spathi")
        self.assertEqual(behaviour.recharge_battery, 5.0)
        self.assertEqual((behaviour.band.near, behaviour.band.far), (500.0, 900.0))
        chmmr = ship_behaviour("Syreen", "Chmmr")
        self.assertEqual(chmmr.specials, ("syreen_chmmr",))
        self.assertIsNone(chmmr.band)
        self.assertIsNone(chmmr.recharge_battery)

    def test_the_utwig_entry_flies_its_earlier_rules_against_the_chmmr_and_spathi(self):
        # Its Chmmr rule is a whole-rule special, and against the Spathi it
        # keeps the earlier band (approach_range to the bolts' reach).
        self.assertIs(strategies.STRATEGIES["Utwig"], strategies.shared)
        for opponent in ("Chmmr", "Spathi"):
            with self.subTest(opponent=opponent):
                earlier = run_match(ArenaMatch("Utwig", opponent, 1, "earlier"), frames=240)
                moved = run_match(ArenaMatch("Utwig", opponent, 1), frames=240)
                self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_utwig_holds_inside_its_bolts_reach(self):
        band = ship_behaviour("Utwig", "Orz").band
        self.assertEqual((band.near, band.far, band.back_off), (500.0, 900.0, True))
        self.assertEqual(ship_behaviour("Utwig", "Spathi").band.far, 1080.0)
        self.assertEqual(ship_behaviour("Utwig", "Shofixti").specials, ("shield_glory",))

    def test_the_yehat_entry_flies_its_earlier_chmmr_rule(self):
        self.assertIs(strategies.STRATEGIES["Yehat"], strategies.shared)
        earlier = run_match(ArenaMatch("Yehat", "Chmmr", 1, "earlier"), frames=240)
        moved = run_match(ArenaMatch("Yehat", "Chmmr", 1), frames=240)
        self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_yehat_closes_to_its_guns_except_against_the_shofixti(self):
        band = ship_behaviour("Yehat", "Orz").band
        self.assertEqual((band.near, band.far, band.back_off), (100.0, 700.0, True))
        shofixti = ship_behaviour("Yehat", "Shofixti")
        self.assertEqual((shofixti.band.near, shofixti.band.far), (504.0, 720.0))
        self.assertEqual(shofixti.specials, ("shield_glory",))

    def test_the_zoq_fot_pik_closes_to_its_gun_except_against_the_orz(self):
        self.assertIs(strategies.STRATEGIES["ZoqFotPik"], strategies.shared)
        behaviour = ship_behaviour("ZoqFotPik", "Spathi")
        self.assertEqual((behaviour.band.near, behaviour.band.far), (100.0, 864.0))
        self.assertEqual(behaviour.recharge_battery, 3.0)
        self.assertEqual(ship_behaviour("ZoqFotPik", "Orz").band.near, 604.8)

    def test_the_slylandro_closes_inside_the_lightnings_reach(self):
        band = ship_behaviour("Slylandro", "Spathi").band
        self.assertEqual((band.near, band.far, band.back_off), (150.0, 300.0, True))
        self.assertIsNone(ship_behaviour("Slylandro", "Chmmr").band.outside_reach_margin)
        self.assertEqual(ship_behaviour("Slylandro", "Chmmr").band.far, 850.0)


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
