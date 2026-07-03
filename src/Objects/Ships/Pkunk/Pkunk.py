import src.const as const
from src.entry_styles import EntryTrailStyle
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Pkunk.A1.PkunkA1 import PkunkA1
from src.Objects.Ships.Pkunk.A2.PkunkA2 import PkunkA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS


class Pkunk(SpaceShip):
    REBIRTH_TRAIL_GAP = 5
    REBIRTH_TRAIL_ANGLES = (45, 135, 225, 315)
    REBIRTH_INDICATOR_COLOR = (0, 255, 0)

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.rebirth_count = 0

    @property
    def current_rebirth_chance(self):
        return self.initial_rebirth_chance * (
            self.rebirth_chance_decay**self.rebirth_count
        )

    @property
    def hud_indicator_color(self):
        return self.REBIRTH_INDICATOR_COLOR

    @property
    def hud_indicator_fraction(self):
        return self.current_rebirth_chance

    @property
    def hud_indicator_size(self):
        return SHIP_DEFINITIONS[self.name].circle_size

    @property
    def hud_indicator_gap(self):
        return SHIP_DEFINITIONS[self.name].circle_gap

    def attempt_rebirth(self):
        if self.rng.random() >= self.current_rebirth_chance:
            return False

        self.rebirth_count += 1
        return True

    def complete_rebirth(self):
        self.current_hp = self.max_hp
        self.current_energy = self.max_energy
        self.energy_timer = 0
        self.currently_alive = True
        self.cloaked = False
        self.trackable = True
        self.can_move = True
        self.can_collide = True
        self.reset_controls()
        self.reset_limpets()
        if self.audio_service is not None:
            self.audio_service.play_effect(
                const.source_path("Objects/Ships/Pkunk/Pkunk-rebirth.wav")
            )

    def on_round_started(self):
        if self.reset_rebirth_chance:
            self.rebirth_count = 0

    def rebirth_entry_trail_style(self):
        return EntryTrailStyle(
            angles=self.REBIRTH_TRAIL_ANGLES,
            spacing=max(self.size) + self.REBIRTH_TRAIL_GAP,
        )

    def plan_action1(self):
        definition = ABILITY_DEFINITIONS["PkunkA1"]
        return self.validate_action(
            1,
            lambda ship: [
                PkunkA1(ship, location, direction)
                for location, direction in zip(
                    definition.gun_locations, definition.gun_directions
                )
            ],
        )

    def plan_action2(self):
        if not self.can_action2() or self.current_energy >= self.max_energy:
            return ActionPlan.invalid(2)
        insult = PkunkA2(self)
        energy_change = min(insult.ENERGY_GAIN, self.max_energy - self.current_energy)
        return self.prepare_action_plan(
            2,
            energy_change=energy_change,
            side_effects=(insult.play_insult,),
            use_first_object_sound=False,
        )
