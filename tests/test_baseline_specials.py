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
    "Thraddash": ("Chmmr", ("thraddash_dodge_burn", "thraddash_trail", "thraddash_chmmr"), (), None),
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
        # The Umgah and Vux run from their table entries, each a whole-rule
        # special, fight for fight as their earlier rules; the Thraddash,
        # split into the schema, still does.
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

    def test_the_utwig_entry_flies_its_earlier_rule_against_the_chmmr(self):
        # Its Chmmr rule is a whole-rule special.
        self.assertIs(strategies.STRATEGIES["Utwig"], strategies.shared)
        earlier = run_match(ArenaMatch("Utwig", "Chmmr", 1, "earlier"), frames=240)
        moved = run_match(ArenaMatch("Utwig", "Chmmr", 1), frames=240)
        self.assertEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_utwig_holds_inside_its_bolts_reach(self):
        band = ship_behaviour("Utwig", "Mycon").band
        self.assertEqual((band.near, band.far, band.back_off), (500.0, 900.0, True))
        spathi = ship_behaviour("Utwig", "Spathi").band
        self.assertEqual((spathi.near, spathi.far, spathi.back_off), (756.0, 1080.0, False))
        self.assertEqual(ship_behaviour("Utwig", "Shofixti").specials, ("shield_glory",))

    def test_the_yehat_flies_its_table_defaults_against_the_chmmr(self):
        # Pass 3 dropped the earlier Chmmr rule.
        self.assertIs(strategies.STRATEGIES["Yehat"], strategies.shared)
        chmmr = ship_behaviour("Yehat", "Chmmr")
        self.assertEqual(chmmr.specials, ())
        self.assertEqual(chmmr.band, ship_behaviour("Yehat").band)
        earlier = run_match(ArenaMatch("Yehat", "Chmmr", 1, "earlier"), frames=240)
        moved = run_match(ArenaMatch("Yehat", "Chmmr", 1), frames=240)
        self.assertNotEqual(dataclasses.replace(earlier, variant="default"), moved)

    def test_the_yehat_closes_to_its_guns_except_against_the_shofixti(self):
        band = ship_behaviour("Yehat", "Orz").band
        self.assertEqual((band.near, band.far, band.back_off), (100.0, 700.0, True))
        shofixti = ship_behaviour("Yehat", "Shofixti")
        self.assertIsNone(shofixti.band)
        self.assertEqual(shofixti.specials, ("shield_glory",))

    def test_the_zoq_fot_pik_closes_to_its_gun_except_against_the_orz(self):
        self.assertIs(strategies.STRATEGIES["ZoqFotPik"], strategies.shared)
        behaviour = ship_behaviour("ZoqFotPik", "Spathi")
        self.assertEqual((behaviour.band.near, behaviour.band.far), (100.0, 864.0))
        self.assertEqual(behaviour.recharge_battery, 3.0)
        self.assertEqual(ship_behaviour("ZoqFotPik", "Orz").band.near, 604.8)

    def test_the_thraddash_holds_a_predictive_band_except_where_the_earlier_one_paid(self):
        band = ship_behaviour("Thraddash", "Pkunk").band
        self.assertEqual((band.near, band.far, band.back_off), (700.0, 1200.0, False))
        for opponent in ("Mmrnmrhm", "Orz", "Spathi", "Thraddash", "Vux", "ZoqFotPik"):
            with self.subTest(opponent=opponent):
                earlier = ship_behaviour("Thraddash", opponent)
                self.assertEqual((earlier.band.near, earlier.band.far, earlier.band.back_off), (900.0, 1400.0, True))
                self.assertEqual(earlier.weapons[0].range_fraction, 1400.0 / 1620.0)

    def test_the_slylandro_closes_inside_the_lightnings_reach(self):
        band = ship_behaviour("Slylandro", "Spathi").band
        self.assertEqual((band.near, band.far, band.back_off), (150.0, 300.0, True))
        self.assertIsNone(ship_behaviour("Slylandro", "Chmmr").band.outside_reach_margin)
        self.assertEqual(ship_behaviour("Slylandro", "Chmmr").band.far, 850.0)


class PerOpponentOverrideTests(unittest.TestCase):
    def test_every_override_names_a_ship_and_a_table_field(self):
        from src.training.baseline.arena import SHIP_TYPE_CATALOG_ORDER
        from src.training.baseline.behaviours import SHIP_BEHAVIOURS

        fields = {field.name for field in dataclasses.fields(ShipBehaviour)} - {"opponents"}
        for ship, entry in SHIP_BEHAVIOURS.items():
            for opponent, overrides in entry.opponents.items():
                with self.subTest(ship=ship, opponent=opponent):
                    self.assertIn(opponent, SHIP_TYPE_CATALOG_ORDER)
                    self.assertTrue(overrides)
                    self.assertLessEqual(set(overrides), fields)

    def test_the_kzer_za_holds_its_band_only_against_eight_opponents(self):
        from src.training.baseline.arena import SHIP_TYPE_CATALOG_ORDER

        banded = {"Arilou", "Chenjesu", "Earthling", "Ilwrath", "KohrAh", "Umgah", "Vux", "Yehat"}
        for opponent in SHIP_TYPE_CATALOG_ORDER:
            with self.subTest(opponent=opponent):
                band = ship_behaviour("KzerZa", opponent).band
                if opponent in banded:
                    self.assertEqual((band.near, band.far, band.inside), (0.0, 1000.0, "face"))
                else:
                    self.assertIsNone(band)

    def test_the_pkunk_band_moves_per_opponent_and_keeps_its_wide_aim(self):
        edges = lambda opponent: (lambda b: (b.near, b.far))(ship_behaviour("Pkunk", opponent).band)
        self.assertEqual(edges("Mycon"), (150.0, 350.0))
        self.assertEqual(edges("Spathi"), (195.0, 455.0))
        self.assertEqual(edges("Syreen"), (105.0, 245.0))
        self.assertEqual(ship_behaviour("Pkunk", "Syreen").weapons[0].tolerance, 45.0)
        # The Arilou flies its base band against the retuned Pkunk.
        self.assertIs(ship_behaviour("Arilou", "Pkunk").band, ship_behaviour("Arilou").band)

    def test_a_band_override_keeps_the_opponents_other_overrides(self):
        arilou = ship_behaviour("Arilou", "Chmmr")
        self.assertEqual(arilou.specials, ("arilou_chmmr",))
        self.assertEqual((arilou.band.near, arilou.band.far), (140.0, 280.0))
        utwig = ship_behaviour("Utwig", "Shofixti")
        self.assertEqual(utwig.specials, ("shield_glory",))
        self.assertIsNone(utwig.band)

    def test_ships_without_a_band_hold_one_at_their_reach_against_some(self):
        self.assertIsNone(ship_behaviour("Mycon", "Spathi").band)
        mycon = ship_behaviour("Mycon", "Shofixti").band
        self.assertEqual((mycon.near, mycon.far, mycon.back_off), (2882.88, 4118.4, False))
        druuge = ship_behaviour("Druuge", "Vux").band
        self.assertEqual((druuge.near, druuge.far, druuge.back_off), (1512.0, 2160.0, True))
        self.assertIsNone(ship_behaviour("Supox", "Spathi").band)
        self.assertEqual(ship_behaviour("Supox", "Pkunk").band.far, 1080.0)

    def test_a_band_override_changes_the_fight(self):
        # Against the Slylandro the Utwig drops its band (pass 2): it deals
        # more in the opening seconds than with the band.
        without = run_match(ArenaMatch("Utwig", "Slylandro", 1), frames=240)
        base = dataclasses.replace(ship_behaviour("Utwig"), opponents={})
        banded = run_match(ArenaMatch("Utwig", "Slylandro", 1, entry=base), frames=240)
        self.assertGreater(without.damage_dealt, banded.damage_dealt)


class MovementTakenTests(unittest.TestCase):
    def test_a_special_that_takes_the_movement_skips_the_band(self):
        from tests.test_baseline_strategy import _rule_context, _seen

        def run(taken):
            context = _rule_context("Thraddash", enemy=_seen(0.0, 400.0))
            context.behaviour = dataclasses.replace(
                context.behaviour,
                specials=(),
                band=Band(900.0, 1400.0, back_off=True),
                weapons=(),
                opponents={},
            )
            context.movement_taken = taken
            strategies.shared(context)
            return context

        self.assertIn("band_near", run(False).memory.touched_sides)
        skipped = run(True)
        self.assertNotIn("band_near", skipped.memory.touched_sides)
        self.assertIsNone(skipped.plan.turn)


class FireShareTests(unittest.TestCase):
    def test_the_gate_is_always_open_at_a_full_share(self):
        from src.training.baseline.engine import fire_gate_open

        self.assertTrue(all(fire_gate_open(d, b, 1.0) for d in range(0, 2000, 37) for b in (-90.0, 0.0, 33.3)))

    def test_the_gate_opens_on_about_its_share_and_repeats(self):
        from src.training.baseline.engine import fire_gate_open

        points = [(300.0 + d * 0.7, (d * 7.31) % 360.0 - 180.0) for d in range(4000)]
        for share in (0.33, 0.5, 0.75):
            with self.subTest(share=share):
                opened = [fire_gate_open(d, b, share) for d, b in points]
                self.assertAlmostEqual(sum(opened) / len(opened), share, delta=0.03)
                self.assertEqual(opened, [fire_gate_open(d, b, share) for d, b in points])

    def test_a_fire_share_must_be_a_share(self):
        from src.training.baseline.behaviours import WeaponUse

        for share in (0.0, -0.5, 1.5):
            with self.subTest(share=share):
                with self.assertRaises(ValueError):
                    WeaponUse(fire_share=share)

    def test_a_closed_gate_holds_a_shot_that_would_fire(self):
        from src.training.baseline.behaviours import FIRE_LAUNCH, WeaponUse
        from tests.test_baseline_strategy import _rule_context, _seen

        fired = {}
        for share in (1.0, 1e-9):
            context = _rule_context("Slylandro", enemy=_seen(0.0, 500.0), battery=20.0)
            fired[share] = context.fire_weapon(1, WeaponUse(FIRE_LAUNCH, fire_share=share))
        self.assertEqual(fired, {1.0: True, 1e-9: False})


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
