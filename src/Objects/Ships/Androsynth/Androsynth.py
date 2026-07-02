from dataclasses import dataclass
import math

from src.Battle.collision_geometry import ship_shape_change_blocked
from src.collision_capabilities import ImpactCapabilities
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Androsynth.A1.AndrosynthA1 import AndrosynthA1
from src.Objects.Ships.Androsynth.A2.AndrosynthA2 import AndrosynthA2
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip


@dataclass
class _FormStatus:
    limpet_count: int = 0
    limpet_sprites: tuple | None = None


class Androsynth(SpaceShip):
    BASE = "Base"
    BLAZER = "A2"

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

    @property
    def is_blazer(self):
        return self.form == self.BLAZER

    def plan_action1(self):
        if self.is_blazer:
            return ActionPlan.invalid(1)
        return self.validate_action(1, AndrosynthA1)

    def plan_action2(self):
        if self.is_blazer or not self.can_action2():
            return ActionPlan.invalid(2)
        transform = AndrosynthA2(self)
        return self.prepare_action_plan(
            2,
            # UQM treats the configured cost as an activation threshold; the
            # Blazer pays for itself through negative energy regeneration.
            energy_change=0,
            side_effects=(self._try_transform,),
            launch_sound=transform.launch_sound,
            use_first_object_sound=False,
        )

    def _try_transform(self):
        assets = self._form_assets[self.BLAZER]
        if ship_shape_change_blocked(self, assets.masks, assets.size):
            return False
        self._activate_form(self.BLAZER)
        return True

    def _activate_form(self, form_name):
        definition = SHIP_DEFINITIONS[self.name].forms[form_name]
        assets = self._form_assets[form_name]
        status = self._form_status[form_name]
        entering_blazer = form_name == self.BLAZER

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
        if definition.inertia is not None:
            self.inertia = definition.inertia
        if definition.mass is not None:
            self.mass = definition.mass
        self.impact_capabilities = ImpactCapabilities(
            ramming_damage=definition.collision_damage
        )
        self.limpets_attached = status.limpet_count
        self._apply_active_limpet_penalties()

        self.thrust_timer = 0
        if entering_blazer:
            # Preserve scalar speed while making the inertialess form move in
            # the direction of its unchanged heading immediately.
            speed = math.hypot(*self.velocity)
            angle = math.radians(self.rotation)
            self.velocity = [
                math.sin(angle) * speed,
                -math.cos(angle) * speed,
            ]
            self.collision_velocity = self.velocity.copy()
        else:
            self.collision_velocity = [0.0, 0.0]

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

    def get_active_thrust_angles(self, thrust_ready, turn_left_ready, turn_right_ready):
        if self.is_blazer:
            return [0]
        return super().get_active_thrust_angles(
            thrust_ready, turn_left_ready, turn_right_ready
        )

    def apply_thrust(self, max_thrust, thrust_increment, angle, make_marker):
        return super().apply_thrust(
            max_thrust,
            thrust_increment,
            angle,
            False if self.is_blazer else make_marker,
        )

    def update_timers(self):
        super().update_timers()
        if self.is_blazer and self.current_energy <= 0:
            self.current_energy = 0
            self._activate_form(self.BASE)

    def handle_incoming_special_object_contact(self, special_object, normal):
        if not self.is_blazer:
            return False

        if special_object.name in ("VuxA2", "SyreenCrew"):
            special_object.set_hp(0)
            return True

        if special_object.name != "OrzA3":
            return False

        if not special_object.currently_alive:
            return True

        dot = (
            special_object.velocity[0] * normal[0]
            + special_object.velocity[1] * normal[1]
        )
        special_object.velocity[0] -= 2 * dot * normal[0]
        special_object.velocity[1] -= 2 * dot * normal[1]
        special_object.position = special_object.previous_position.copy()
        special_object.shield_bounce_timer = special_object.SHIELD_BOUNCE_FRAMES
        return True
