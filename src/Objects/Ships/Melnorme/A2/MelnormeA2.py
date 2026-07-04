from dataclasses import replace

import src.const as const
from src.collision_capabilities import ProjectileContactPolicy
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class MelnormeA2(Ability):
    def __init__(self, parent):
        super().__init__("MelnormeA2", parent)
        definition = ABILITY_DEFINITIONS[self.name]
        resource_dir = const.source_path(definition.file_path)
        self.effect_frames = self.resources.animation(
            f"{self.name}-confused-effect",
            tuple(
                resource_dir / f"{self.name}effect{frame:02d}.png"
                for frame in range(definition.effect_frames)
            ),
        )
        self.effect_duration = definition.effect_duration
        # This pulse's eight source frames are distributed across its exact
        # twenty-frame lifetime; unlike ordinary spawned abilities, it does
        # not use the one-tick first-frame presentation extension.
        self.frame_timer = self.frame_delay
        self.area_damage_capabilities = type(self.area_damage_capabilities)(
            targetable=True,
            vulnerable=False,
        )
        self.special_object_collision_capabilities = replace(
            self.special_object_collision_capabilities,
            projectile_contact_policy=ProjectileContactPolicy.TAKE_DAMAGE,
        )
        self.launch_from_gun()

    def should_collide_with_projectile_like(self, other):
        return (
            getattr(other, "type", None) == "projectile"
            and other.player != self.player
        )

    def handle_projectile_contact(self, projectile):
        capabilities = getattr(
            projectile, "special_object_collision_capabilities", None
        )
        damage = (
            projectile.current_damage
            if capabilities is None or capabilities.damages_projectiles
            else 0
        )
        self.set_hp(self.current_hp - damage)
        return True

    def handle_ship_contact(self, ship, normal=None):
        direction = self.rng.choice((-1, 1))
        ship.apply_confused(
            self.effect_frames,
            self.effect_duration,
            turn_direction=direction,
        )
        self.current_hp = 0
        self.currently_alive = False
        return True
