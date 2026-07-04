from dataclasses import dataclass

from src.Battle.collision_geometry import ship_shape_change_blocked
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Mmrnmrhm.A1.MmrnmrhmXFormA1 import MmrnmrhmXFormA1
from src.Objects.Ships.Mmrnmrhm.A1.MmrnmrhmYWingA1 import MmrnmrhmYWingA1
from src.Objects.Ships.Mmrnmrhm.A2.MmrnmrhmA2 import MmrnmrhmA2


@dataclass
class _FormStatus:
    limpet_count: int = 0
    limpet_sprites: tuple | None = None


class Mmrnmrhm(SpaceShip):
    XFORM = "XForm"
    YWING = "YWing"

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        definition = SHIP_DEFINITIONS[ship_name]
        self.form = definition.default_form
        self._form_assets = {
            form_name: self.resources.ship_form(ship_name, form_name)
            for form_name in definition.forms
        }
        self._form_status = {
            form_name: _FormStatus() for form_name in definition.forms
        }
        self._activate_form(self.form)

    def plan_action1(self):
        factory = (
            MmrnmrhmXFormA1.create_beams
            if self.form == self.XFORM
            else MmrnmrhmYWingA1.create_projectiles
        )
        return self.validate_action(1, factory)

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        transform = MmrnmrhmA2(self)
        return self.prepare_action_plan(
            2,
            side_effects=(self._try_transform,),
            launch_sound=transform.launch_sound,
            use_first_object_sound=False,
        )

    def _try_transform(self):
        next_form = self.YWING if self.form == self.XFORM else self.XFORM
        assets = self._form_assets[next_form]
        if ship_shape_change_blocked(self, assets.masks, assets.size):
            return False
        self._activate_form(next_form)
        # UQM clears weapon_counter after a successful transformation, so the
        # outgoing form's primary cooldown does not carry into the new form.
        self.action1_timer = 0
        return True

    def _activate_form(self, form_name):
        definition = SHIP_DEFINITIONS[self.name].forms[form_name]
        assets = self._form_assets[form_name]
        status = self._form_status[form_name]

        self.form = form_name
        self.base_sprites = assets.sprites
        self.sprites = status.limpet_sprites or assets.sprites
        self.masks = assets.masks
        self.size = list(assets.size)
        self.sprite_scale = definition.sprite_scale
        self.energy_regen = definition.energy_regen
        self.energy_wait = definition.energy_wait
        self.max_thrust = definition.max_thrust
        self.thrust_increment = definition.thrust_increment
        self.thrust_wait = definition.thrust_wait
        self.turn_wait = definition.turn_wait
        self.a1_cost = definition.a1_cost
        self.a1_wait = definition.a1_wait
        self.limpets_attached = status.limpet_count
        self._apply_active_limpet_penalties()

    def _apply_active_limpet_penalties(self):
        definition = self._intrinsic_movement_definition()
        count = self.limpets_attached
        self.turn_wait = min(255, definition.turn_wait + count)
        self.thrust_wait = min(255, definition.thrust_wait + count)
        if definition.thrust_increment > 4:
            self.thrust_increment = max(4, definition.thrust_increment - count)
            self.max_thrust = int(
                definition.max_thrust
                * self.thrust_increment
                / definition.thrust_increment
            )
        else:
            self.thrust_increment = definition.thrust_increment
            self.max_thrust = 8 if count else definition.max_thrust

    def _intrinsic_movement_definition(self):
        return SHIP_DEFINITIONS[self.name].forms[self.form]

    def attach_limpet(self):
        super().attach_limpet()
        status = self._form_status[self.form]
        status.limpet_count = self.limpets_attached
        status.limpet_sprites = self.sprites

    def reset_limpets(self):
        if not hasattr(self, "_form_status"):
            return super().reset_limpets()
        for status in self._form_status.values():
            status.limpet_count = 0
            status.limpet_sprites = None
        self._activate_form(self.form)
