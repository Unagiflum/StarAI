from src.Objects.object import PlayerObject, ThrustMarker
import src.const as const
import math
import pygame
import random
from src.Battle.collision_geometry import ship_rotation_blocked
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    ShipImpactResult,
    PhysicalCollisionCapabilities,
    ImpactCapabilities,
    DurabilityCapabilities,
)
from src.Objects.Ships.catalog import SHIP_DEFINITIONS, SHIPS_DATA
from src.Objects.Ships.action_transaction import ActionOutput, ActionPlan, ActionResult
from src.resources import default_assets
from src.entry_styles import STANDARD_ENTRY_TRAIL

CONTROL_STATE_ATTRIBUTES = {
    "thrust": "thrust_active",
    "turn_left": "turn_left_active",
    "turn_right": "turn_right_active",
    "action1": "action1_active",
    "action2": "action2_active",
}


class SpaceShip(PlayerObject):
    action_factories = {}

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        # Get ship-specific data from cached data
        ship_definition = SHIP_DEFINITIONS[ship_name]
        sprite_location = const.source_path(ship_definition.sprite_path)

        # Initialize the PlayerObject base class
        super().__init__(
            name=ship_name,
            sprite_location=sprite_location,
            size=[0, 0],
            player=player_num,
            sprite_scale=ship_definition.sprite_scale,
        )
        self.resources = resources or default_assets()
        self.rng = random
        # A battle composition root may bind this after fleet construction.
        # None preserves legacy constructors and their Ability.sound_enabled fallback.
        self.audio_service = audio_service
        self.collision_capabilities = CollisionCapabilities(CollisionRole.SHIP)
        self.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(bounces_on_immovable=True)
        self.impact_capabilities = ImpactCapabilities()
        self.durability_capabilities = DurabilityCapabilities(
            immune_to_psychic=ship_definition.immune_to_psychic
        )

        assets = self.resources.ship(ship_name)
        self.size = list(assets.size)
        self.sprites = assets.sprites
        self.masks = assets.masks

        # Ship-specific attributes
        self.ship_type = ship_definition.ship_type
        self.cost = ship_definition.cost
        self.max_hp = ship_definition.max_hp
        self.start_hp = ship_definition.start_hp
        self.max_energy = ship_definition.max_energy
        self.start_energy = ship_definition.start_energy
        self.energy_regen = ship_definition.energy_regen
        self.energy_wait = ship_definition.energy_wait
        self.max_thrust = ship_definition.max_thrust
        self.thrust_increment = ship_definition.thrust_increment
        self.thrust_wait = ship_definition.thrust_wait
        self.turn_wait = ship_definition.turn_wait
        self.a1_cost = ship_definition.a1_cost
        self.a2_cost = ship_definition.a2_cost
        self.a3_cost = ship_definition.a3_cost
        self.a1_wait = ship_definition.a1_wait
        self.a2_wait = ship_definition.a2_wait
        self.a3_wait = ship_definition.a3_wait
        self.mass = ship_definition.mass
        self.inertia = ship_definition.inertia
        self.initial_rebirth_chance = ship_definition.initial_rebirth_chance
        self.rebirth_chance_decay = ship_definition.rebirth_chance_decay
        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self.planet_contacts = set()

        self.current_hp = ship_definition.start_hp
        self.current_energy = ship_definition.start_energy
        self.energy_timer = 0

        # Timers
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0

        self.can_die = True
        self._active_damage_shield = None
        self.cloaked = False
        self.trackable = True
        self.boarded_marines = []
        self.input_pressed_frames = {}
        self.newly_pressed_controls = set()
        self.released_controls = set()
        self.reset_controls()
        self.camera_freeze_timer = 0
        self.frozen_camera_position = [0.0, 0.0]
        self.battles_fought = 0
        self.limpets_attached = 0
        self.base_sprites = self.sprites

    def initialize_in_battle(self, position, heading):
        self.position = list(position)
        self.previous_position = self.position.copy()
        self.heading = heading % const.SHIP_DIRECTIONS
        self.previous_heading = self.heading
        self.rotation = self.heading * const.TURN_ANGLE
        self.velocity = [0.0, 0.0]
        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self.planet_contacts.clear()
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0
        self.boarded_marines.clear()
        self.reset_controls()
        self.in_battle = True
        self.battles_fought += 1

    def attach_limpet(self):
        ship_definition = self._intrinsic_movement_definition()
        start_thrust_inc = ship_definition.thrust_increment
        start_max_thrust = ship_definition.max_thrust
        start_turn_wait = ship_definition.turn_wait
        start_thrust_wait = ship_definition.thrust_wait

        self.limpets_attached += 1

        # Calculate new stats based on number of limpets
        self.turn_wait = min(255, start_turn_wait + self.limpets_attached)
        self.thrust_wait = min(255, start_thrust_wait + self.limpets_attached)

        if start_thrust_inc > 4:
            self.thrust_increment = max(4, start_thrust_inc - self.limpets_attached)
            self.max_thrust = int(
                start_max_thrust * self.thrust_increment / start_thrust_inc
            )
        else:
            self.thrust_increment = start_thrust_inc
            self.max_thrust = 8

        # Draw limpet onto the ship's sprites
        limpet_assets = self.resources.ability("VuxA2")
        if not limpet_assets.sprites:
            return

        limpet_sprite = limpet_assets.sprites[0]

        # Find a random valid point on the first base sprite with alpha > 200
        base_sprite = self.base_sprites[0]
        width, height = base_sprite.get_size()
        center_x, center_y = width / 2, height / 2

        spot_offset_x = 0
        spot_offset_y = 0
        found = False

        for _ in range(50):
            rx = self.rng.randint(0, width - 1)
            ry = self.rng.randint(0, height - 1)
            color = base_sprite.get_at((rx, ry))
            if color.a > 200:
                spot_offset_x = rx - center_x
                spot_offset_y = ry - center_y
                found = True
                break

        if not found:
            return

        # Draw the limpet at the rotated offset for all 64 directions
        new_sprites = []
        for heading in range(const.TOTAL_SPRITE_DIRECTIONS):
            angle = math.radians(heading * const.TOTAL_SPRITE_STEP)
            # Rotate offset clockwise (since heading goes clockwise 0=up, 16=right)
            rx = spot_offset_x * math.cos(angle) - spot_offset_y * math.sin(angle)
            ry = spot_offset_x * math.sin(angle) + spot_offset_y * math.cos(angle)

            # Blit onto current sprite so multiple limpets stack
            current_sprite = self.sprites[heading].copy()
            dest_x = current_sprite.get_width() / 2 + rx - limpet_sprite.get_width() / 2
            dest_y = (
                current_sprite.get_height() / 2 + ry - limpet_sprite.get_height() / 2
            )

            current_sprite.blit(limpet_sprite, (int(dest_x), int(dest_y)))
            new_sprites.append(current_sprite)

        self.sprites = tuple(new_sprites)

    def reset_limpets(self):
        if self.limpets_attached == 0:
            return

        self.limpets_attached = 0
        self.sprites = self.base_sprites

        # Reset stats
        ship_definition = self._intrinsic_movement_definition()
        self.max_thrust = ship_definition.max_thrust
        self.thrust_increment = ship_definition.thrust_increment
        self.thrust_wait = ship_definition.thrust_wait
        self.turn_wait = ship_definition.turn_wait

    def _intrinsic_movement_definition(self):
        """Return unmodified movement stats for status-effect calculations."""
        return SHIP_DEFINITIONS[self.name]

    def reset_controls(self):
        for attribute in CONTROL_STATE_ATTRIBUTES.values():
            setattr(self, attribute, False)
        self.input_pressed_frames.clear()
        self.newly_pressed_controls.clear()
        self.released_controls.clear()

    def rebirth_entry_trail_style(self):
        return STANDARD_ENTRY_TRAIL

    def set_control_state(self, control_name, pressed, frame_id=None):
        attribute = CONTROL_STATE_ATTRIBUTES[control_name]
        was_active = getattr(self, attribute)
        if was_active == pressed:
            return

        setattr(self, attribute, pressed)
        if pressed:
            self.input_pressed_frames[control_name] = frame_id
            self.newly_pressed_controls.add(control_name)
        else:
            self.input_pressed_frames.pop(control_name, None)
            self.released_controls.add(control_name)

    def process_controls(self, frame_id=None):
        new_objects = []
        self.update_timers()
        self.previous_heading = getattr(self, "heading", 0)

        # Handle movement based on active states
        turn_left_ready = self.control_ready("turn_left", frame_id)
        turn_right_ready = self.control_ready("turn_right", frame_id)
        thrust_ready = self.control_ready("thrust", frame_id)
        action1_ready = self.control_ready("action1", frame_id)
        action2_ready = self.control_ready("action2", frame_id)

        if self.turn_left_active and turn_left_ready and self.turn_input_enabled():
            self.turn_left()
        if self.turn_right_active and turn_right_ready and self.turn_input_enabled():
            self.turn_right()
        thrust_angles = self.get_active_thrust_angles(
            thrust_ready, turn_left_ready, turn_right_ready
        )
        if thrust_angles and self.can_thrust():
            self.thrust_timer = const.cooldown_frames(self.thrust_wait)
            for thrust_angle in thrust_angles:
                marker = self.apply_thrust(
                    self.max_thrust,
                    self.thrust_increment,
                    thrust_angle,
                    not self.cloaked,
                )
                if marker:
                    new_objects.append(marker)

        # Input processing consumes typed transactions directly. The
        # perform_action* methods below remain compatibility wrappers.
        action1_active = self.action1_active or "action1" in self.newly_pressed_controls
        action2_active = self.action2_active or "action2" in self.newly_pressed_controls
        if action1_active and action2_active and (action1_ready or action2_ready):
            result = self.commit_action(self._select_action_plan(3))
            new_objects.extend(result.spawned_objects)
            combination_handled = self.handles_combined_action()
            if not combination_handled:
                if action1_active and action1_ready:
                    result = self.commit_action(self._select_action_plan(1))
                    new_objects.extend(result.spawned_objects)
                if action2_active and action2_ready:
                    result = self.commit_action(self._select_action_plan(2))
                    new_objects.extend(result.spawned_objects)
        else:
            if action1_active and action1_ready:
                result = self.commit_action(self._select_action_plan(1))
                new_objects.extend(result.spawned_objects)
            if action2_active and action2_ready:
                result = self.commit_action(self._select_action_plan(2))
                new_objects.extend(result.spawned_objects)

        if "action1" in self.released_controls:
            self.perform_action1_release()

        self.newly_pressed_controls.clear()
        self.released_controls.clear()
        return new_objects

    @staticmethod
    def normalize_action_result(result):
        if isinstance(result, ActionResult):
            return list(result.spawned_objects) if result.valid else []
        if not result:
            return []
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

    def _select_action_plan(self, action_number: int) -> ActionPlan:
        if action_number == 1:
            return self.plan_action1()
        if action_number == 2:
            return self.plan_action2()
        if action_number == 3:
            return self.plan_action3()
        raise ValueError(f"Unsupported action number: {action_number}")

    def turn_input_enabled(self):
        return True

    def get_active_thrust_angles(self, thrust_ready, turn_left_ready, turn_right_ready):
        if self.thrust_active and thrust_ready:
            return [0]
        return []

    def control_ready(self, control_name, frame_id):
        if control_name in self.newly_pressed_controls:
            return True
        if frame_id is None:
            return True

        pressed_frame = self.input_pressed_frames.get(control_name)
        if pressed_frame is None:
            return True

        return frame_id - pressed_frame >= const.INPUT_REPEAT_DELAY_FRAMES

    def on_ship_impact(self, other, impact):
        """Return optional behavior for a physical collision with another ship."""
        return ShipImpactResult()

    def on_elastic_bounce(self, other):
        """React after elastic collision physics updates this object's velocity."""
        return None

    def is_alive(self):
        return self.currently_alive and self.current_hp > 0

    def damage_shield_is_active(self):
        shield = getattr(self, "_active_damage_shield", None)
        return bool(
            shield is not None and shield.currently_alive and shield.blocks_damage
        )

    def destroy_boarded_marines(self):
        """Destroy every hostile unit currently carried by this ship."""
        for marine in tuple(self.boarded_marines):
            on_host_self_destruct = getattr(marine, "on_host_self_destruct", None)
            if on_host_self_destruct is not None:
                on_host_self_destruct()
        self.boarded_marines.clear()

    def take_damage(self, damage, *, shieldable=True, non_lethal=False):
        """Apply hull/crew damage and return the amount actually taken.

        Non-damage effects should not use this method. Exceptional damage that
        explicitly bypasses shields can pass ``shieldable=False``.
        """
        damage = max(0, damage)
        if damage <= 0 or self.current_hp <= 0:
            return 0
        if shieldable and self.damage_shield_is_active():
            return 0

        previous_hp = self.current_hp
        min_hp = 1 if non_lethal else 0
        self.current_hp = max(min_hp, self.current_hp - damage)
        return previous_hp - self.current_hp

    def update(self):
        self.previous_position = self.position.copy()
        self.update_physics()
        return True

    def update_physics(self):
        if not self.inertia:
            if self.collision_velocity != [0.0, 0.0]:
                self.velocity = self.collision_velocity.copy()
                self.collision_velocity = [0.0, 0.0]
            else:
                if self.accumulated_impulses != [0.0, 0.0]:
                    self.velocity = self.accumulated_impulses.copy()
                elif self.thrust_timer == 0 and not self.thrust_active:
                    self.velocity = [0.0, 0.0]
            self.accumulated_impulses = [0.0, 0.0]

        super().update_physics()
        self.apply_speed_limit()

    def can_thrust(self):
        return self.thrust_timer == 0

    def can_turn(self):
        return self.turn_timer == 0

    def can_action1(self):
        return self.action1_timer == 0 and self.current_energy >= self.a1_cost

    def can_action2(self):
        return self.action2_timer == 0 and self.current_energy >= self.a2_cost

    def can_action3(self):
        return self.action3_timer == 0 and self.current_energy >= self.a3_cost

    @property
    def camera_position(self):
        if self.camera_freeze_timer > 0:
            return self.frozen_camera_position
        return self.position

    def update_timers(self):
        if self.camera_freeze_timer > 0:
            self.camera_freeze_timer -= 1
        if self.thrust_timer > 0:
            self.thrust_timer -= 1
        if self.turn_timer > 0:
            self.turn_timer -= 1
        if self.action1_timer > 0:
            self.action1_timer -= 1
        if self.action2_timer > 0:
            self.action2_timer -= 1
        if self.action3_timer > 0:
            self.action3_timer -= 1
        if not self.inertia and self.thrust_timer == 0 and not self.thrust_active:
            self.velocity = [0.0, 0.0]

        self.energy_timer += 1
        if self.energy_timer > self.energy_wait:

            self.energy_timer = 0
            if self.current_energy < self.max_energy:
                self.current_energy = min(
                    self.max_energy, self.current_energy + self.energy_regen
                )

    def turn_left(self):
        if self.can_turn():
            old_heading = self.heading
            old_rotation = self.rotation
            self.heading = (self.heading - 1) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE
            if self.rotation_would_overlap():
                self.heading = old_heading
                self.rotation = old_rotation
                return
            self.turn_timer = const.cooldown_frames(self.turn_wait)

    def turn_right(self):
        if self.can_turn():
            old_heading = self.heading
            old_rotation = self.rotation
            self.heading = (self.heading + 1) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE
            if self.rotation_would_overlap():
                self.heading = old_heading
                self.rotation = old_rotation
                return
            self.turn_timer = const.cooldown_frames(self.turn_wait)

    def rotation_would_overlap(self):
        return ship_rotation_blocked(self)

    def get_collision_mask(self):
        masks = getattr(self, "masks", None)
        if not masks:
            return None
        return masks[const.heading_to_sprite_index(self.heading)]

    def validate_action(self, action_number, factory=None) -> ActionPlan:
        """Prepare a common action without changing ship state."""
        if not getattr(self, f"can_action{action_number}")():
            return ActionPlan.invalid(action_number)

        raw_result = factory(self) if factory else None
        return self.prepare_action_plan(action_number, raw_result)

    def prepare_action_plan(
        self,
        action_number,
        raw_result=None,
        *,
        energy_change=None,
        crew_change=0,
        side_effects=(),
        launch_sound=None,
        use_first_object_sound=True,
        output=None,
    ) -> ActionPlan:
        """Build a plan after ship-specific validation has succeeded."""
        cost = getattr(self, f"a{action_number}_cost")
        wait = getattr(self, f"a{action_number}_wait")
        objects = tuple(self.normalize_action_result(raw_result))
        if output is None:
            if isinstance(raw_result, (list, tuple)):
                output = ActionOutput.MANY
            elif raw_result is None:
                output = ActionOutput.NONE
            else:
                output = ActionOutput.SINGLE
        return ActionPlan(
            action_number=action_number,
            valid=True,
            spawned_objects=objects,
            energy_change=-cost if energy_change is None else energy_change,
            crew_change=crew_change,
            cooldown_frames=const.cooldown_frames(wait),
            cooldown_committed=True,
            side_effects=tuple(side_effects),
            launch_sound=launch_sound,
            use_first_object_sound=use_first_object_sound,
            output=output,
        )

    def commit_action(self, plan: ActionPlan) -> ActionResult:
        """Apply a validated plan and return its ordered typed outcome."""
        if not plan.valid:
            return ActionResult.invalid()

        self.current_energy += plan.energy_change
        self.current_hp += plan.crew_change
        cooldown_action = None
        if plan.cooldown_committed:
            cooldown_action = plan.action_number
            setattr(self, f"action{plan.action_number}_timer", plan.cooldown_frames)

        for effect in plan.side_effects:
            effect()

        sound = plan.launch_sound
        if sound is None and plan.use_first_object_sound:
            for action_object in plan.spawned_objects:
                sound = getattr(action_object, "launch_sound", None)
                if sound:
                    break
        if sound:
            sound.play()

        return ActionResult(
            valid=True,
            spawned_objects=plan.spawned_objects,
            energy_change=plan.energy_change,
            crew_change=plan.crew_change,
            cooldown_action=cooldown_action,
            cooldown_frames=plan.cooldown_frames if cooldown_action else 0,
            side_effects=plan.side_effects,
            launch_sound_played=bool(sound),
            output=plan.output,
        )

    def execute_action_result(self, action_number, factory=None) -> ActionResult:
        """Validate and commit a common action, returning its typed result."""
        return self.commit_action(self.validate_action(action_number, factory))

    def execute_action(self, action_number, factory=None):
        """Compatibility wrapper returning the historical raw action value."""
        return self.execute_action_result(action_number, factory).compatibility_value()

    def plan_action1(self):
        return self.validate_action(1, self.action_factories.get(1))

    def plan_action2(self):
        return self.validate_action(2, self.action_factories.get(2))

    def plan_action3(self):
        return ActionPlan.invalid(3)

    def handles_combined_action(self):
        return False

    def perform_action1(self):
        return self.commit_action(self.plan_action1()).compatibility_value()

    def perform_action1_release(self):
        return None

    def perform_action2(self):
        return self.commit_action(self.plan_action2()).compatibility_value()

    def perform_action3(self):
        result = self.commit_action(self.plan_action3())
        return result.compatibility_value(), self.handles_combined_action()

    def set_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index

        return self.sprites[interpolated_sprite_index(self, interp_t)]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        sprite = self.set_sprite(interp_t)

        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)

        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        # Calculate screen position with translation
        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)

        # Draw the ship at all potential wrap-around positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                # Only draw if the position would be visible
                if (
                    -scaled_rect.width
                    <= pos_x
                    <= const.SCREEN_HEIGHT + scaled_rect.width
                    and -scaled_rect.height
                    <= pos_y
                    <= const.SCREEN_HEIGHT + scaled_rect.height
                ):
                    screen.blit(
                        scaled_sprite,
                        (
                            const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                            pos_y - scaled_rect.height // 2,
                        ),
                    )
