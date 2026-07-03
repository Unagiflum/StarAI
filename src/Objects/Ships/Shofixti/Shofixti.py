from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.Shofixti.A1.ShofixtiA1 import ShofixtiA1
from src.Objects.Ships.Shofixti.A2.ShofixtiA2 import ShofixtiA2


class Shofixti(SpaceShip):
    action_factories = {1: ShofixtiA1}
    SAFE = 0
    CAUTION = 1
    ARMED = 2
    ARMING_INDICATOR_COLORS = (
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shofixti_self_destruct = False
        self.shofixti_arming_stage = self.SAFE

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.shofixti_self_destruct = False
        self.shofixti_arming_stage = self.SAFE

    @property
    def hud_indicator_color(self):
        stage = getattr(self, "shofixti_arming_stage", self.SAFE)
        return self.ARMING_INDICATOR_COLORS[stage]

    @property
    def hud_indicator_size(self):
        return SHIP_DEFINITIONS[self.name].circle_size

    @property
    def hud_indicator_gap(self):
        return SHIP_DEFINITIONS[self.name].circle_gap

    def control_ready(self, control_name, frame_id):
        if control_name == "action2":
            return control_name in self.newly_pressed_controls
        return super().control_ready(control_name, frame_id)

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        if self.shofixti_arming_stage < self.ARMED:
            return ActionPlan(
                action_number=2,
                valid=True,
                resets_energy_wait=False,
                side_effects=(self._advance_self_destruct_arming,),
            )
        explosion = ShofixtiA2(self)
        side_effects = ()
        crew_change = 0
        if self.in_battle:
            crew_change = -self.current_hp
            side_effects = (self._mark_self_destruct,)
        return self.prepare_action_plan(
            2,
            explosion,
            crew_change=crew_change,
            side_effects=side_effects,
        )

    def _advance_self_destruct_arming(self):
        self.shofixti_arming_stage = min(
            self.ARMED,
            self.shofixti_arming_stage + 1,
        )

    def _mark_self_destruct(self):
        self.shofixti_self_destruct = True
        self.destroy_boarded_marines()
