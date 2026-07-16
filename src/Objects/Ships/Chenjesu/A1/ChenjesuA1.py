import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.training import event_ledger


class _ChenjesuCrystal:
    def should_collide_with_projectile_like(self, other):
        if getattr(other, "name", None) == "ChenjesuA2":
            return True
        return not (
            other.player == self.player
            and isinstance(other, (ChenjesuA1, ChenjesuA1Shard))
        )


class ChenjesuA1(_ChenjesuCrystal, Ability):
    def __init__(self, parent):
        super().__init__("ChenjesuA1", parent)
        self.expiration_timer = float("inf")
        self.spawned_objects = []
        self.fragmented = False
        self.fragment_pending = False
        directory = const.source_path(ABILITY_DEFINITIONS[self.name].file_path)
        self.break_sound = self.audio_service.load_effect(
            directory / "ChenjesuA1break.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        self.launch_from_gun()

    def request_fragment(self):
        if not self.fragmented and self.is_alive():
            self.fragment_pending = True

    def finalize_collision_frame(self):
        if self.fragment_pending and self.is_alive():
            self.fragment()

    def fragment(self):
        if self.fragmented or not self.is_alive():
            return
        self.fragmented = True
        self.fragment_pending = False
        definition = ABILITY_DEFINITIONS[self.name]
        angle_step = 360.0 / definition.fragment_count
        self.spawned_objects = [
            ChenjesuA1Shard(self, index * angle_step)
            for index in range(definition.fragment_count)
        ]
        if self.break_sound:
            self.break_sound.play()
        self.current_hp = 0
        self.currently_alive = False
        self._clear_parent_reference()

    def drain_spawned_objects(self):
        spawned = self.spawned_objects
        self.spawned_objects = []
        return spawned

    def on_destroyed(self):
        self._clear_parent_reference()

    def stop_and_track(self):
        self.fragment()

    def _clear_parent_reference(self):
        if getattr(self.parent, "active_a1", None) is self:
            self.parent.active_a1 = None


class ChenjesuA1Shard(_ChenjesuCrystal, Ability):
    def __init__(self, source, direction):
        super().__init__("ChenjesuA1", source.parent)
        event_ledger.inherit_credit(self, source)
        definition = ABILITY_DEFINITIONS["ChenjesuA1"]
        self.name = "ChenjesuA1Shard"
        self.projectile_name = self.name
        self.start_hp = definition.fragment_hp
        self.current_hp = self.start_hp
        self.hp_array = (self.start_hp,)
        self.current_damage = definition.fragment_damage
        self.damages = (self.current_damage,)
        self.speed = definition.fragment_speed
        self.life_time = definition.fragment_life_time
        self._duration = self.life_time
        self.expiration_timer = self.life_time
        self.inertia = False
        self.death_animation = ()
        self.position = list(source.position)
        self.previous_position = self.position.copy()
        self.rotation = direction % 360
        self.heading = 0
        angle = math.radians(self.rotation)
        self.velocity = [
            math.sin(angle) * self.speed,
            -math.cos(angle) * self.speed,
        ]

        directory = const.source_path(definition.file_path)
        shard = self.resources.image(
            directory / "ChenjesuA1shard.png",
            with_mask=True,
        )
        self.shard_sprite = shard.image
        self.shard_mask = shard.mask or pygame.mask.from_surface(shard.image)
        self.size = list(self.shard_sprite.get_size())
        self.launch_sound = None

    def get_sprite(self, interp_t=0.0):
        return self.shard_sprite

    def get_collision_mask(self):
        return self.shard_mask
