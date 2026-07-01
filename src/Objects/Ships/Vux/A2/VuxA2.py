from dataclasses import replace
import math

import src.const as const
from src.Objects.Ships.ability import (
    Ability,
    ABILITIES_DATA,
    SPECIAL_OBJECT_AREA_IMMUNITIES,
)
from src.collision_capabilities import ProjectileContactPolicy


class VuxA2(Ability):
    def __init__(self, parent):
        super().__init__("VuxA2", parent)
        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            immune_to_sources=SPECIAL_OBJECT_AREA_IMMUNITIES,
        )
        self.special_object_collision_capabilities = replace(
            self.special_object_collision_capabilities,
            projectile_contact_policy=ProjectileContactPolicy.FRAGILE,
        )
        self.physical_collision_capabilities = replace(
            self.physical_collision_capabilities,
            is_fragile=True,
        )
        ability_data = ABILITIES_DATA["VuxA2"]

        # Override default sound loading since filenames differ from standard
        if self.audio_service:
            path = const.source_path("Objects/Ships/Vux/A2/")
            self.launch_sound = self.audio_service.load_effect(
                path / "VuxA2Launch.wav", const.SOUND_EFFECT_VOLUME
            )
            self.bite_sound = self.audio_service.load_effect(
                path / "VuxA2Bite.wav", const.SOUND_EFFECT_VOLUME
            )
        else:
            self.bite_sound = None

        self.place_self()

    def place_self(self):
        opponent = getattr(self.parent, "opponent", None)
        has_valid_target = (
            opponent is not None
            and getattr(opponent, "currently_alive", True)
            and getattr(opponent, "current_hp", 1) > 0
            and not getattr(opponent, "cloaked", False)
        )
        if has_valid_target:
            from src.toroidal import wrapped_delta

            muzzle = self.configured_gun_position()
            dx, dy = wrapped_delta(muzzle, opponent.position)
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360
            direction_step = 360 / const.SHIP_DIRECTIONS
            direction = round(target_angle / direction_step) * direction_step
            self.launch_from_gun(launch_direction=direction)
        else:
            self.launch_from_gun(inherit_parent_velocity=False)

    def handle_ship_contact(self, ship, normal=None):
        if hasattr(ship, "attach_limpet"):
            ship.attach_limpet()
        if getattr(self, "bite_sound", None):
            self.bite_sound.play()

        self.current_hp = 0
        self.currently_alive = False
        return True

    def handle_projectile_contact(self, projectile):
        if getattr(projectile, "projectile_name", None) == self.projectile_name:
            return False
        self.set_hp(0)
        return True
