from src.Objects.object import PlayerObject, ThrustMarker
import src.const as const
import math
import pygame
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    ShipImpactResult,
)
from src.Objects.Ships.catalog import SHIP_DEFINITIONS, SHIPS_DATA
from src.resources import default_assets


CONTROL_STATE_ATTRIBUTES = {
    "thrust": "thrust_active",
    "turn_left": "turn_left_active",
    "turn_right": "turn_right_active",
    "action1": "action1_active",
    "action2": "action2_active",
}

class SpaceShip(PlayerObject):
    action_factories = {}

    def __init__(self, ship_name, player_num, resources=None):
        # Get ship-specific data from cached data
        ship_definition = SHIP_DEFINITIONS[ship_name]
        sprite_location = const.source_path(ship_definition.sprite_path)

        # Initialize the PlayerObject base class
        super().__init__(
            name=ship_name,
            sprite_location=sprite_location,
            size=[0,0],
            player=player_num,
            sprite_scale=ship_definition.sprite_scale
        )
        self.resources = resources or default_assets()
        self.collision_capabilities = CollisionCapabilities(CollisionRole.SHIP)
        self.area_damage_capabilities = AreaDamageCapabilities(targetable=True)

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
        self.collision_velocity = [0.0, 0.0]
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
        self.cloaked = False
        self.trackable = True
        self.input_pressed_frames = {}
        self.newly_pressed_controls = set()
        self.released_controls = set()
        self.reset_controls()

    def initialize_in_battle(self, position, heading):
        self.position = list(position)
        self.previous_position = self.position.copy()
        self.heading = heading % const.SHIP_DIRECTIONS
        self.rotation = self.heading * const.TURN_ANGLE
        self.velocity = [0.0, 0.0]
        self.collision_velocity = [0.0, 0.0]
        self.planet_contacts.clear()
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0
        self.reset_controls()
        self.in_battle = True

    def reset_controls(self):
        for attribute in CONTROL_STATE_ATTRIBUTES.values():
            setattr(self, attribute, False)
        self.input_pressed_frames.clear()
        self.newly_pressed_controls.clear()
        self.released_controls.clear()

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
        for thrust_angle in self.get_active_thrust_angles(
                thrust_ready, turn_left_ready, turn_right_ready):
            marker = self.apply_thrust(
                self.max_thrust,
                self.thrust_increment,
                thrust_angle,
                self.can_thrust(),
                not self.cloaked
            )
            if marker:
                new_objects.append(marker)

        # Handle actions based on active states
        action1_active = self.action1_active or "action1" in self.newly_pressed_controls
        action2_active = self.action2_active or "action2" in self.newly_pressed_controls
        if action1_active and action2_active and (action1_ready or action2_ready):
            result, is_valid = self.perform_action3()
            self._add_action_result(new_objects, result)
            if not is_valid:
                if action1_active and action1_ready:
                    self._add_action_result(new_objects, self.perform_action1())
                if action2_active and action2_ready:
                    self._add_action_result(new_objects, self.perform_action2())
        else:
            if action1_active and action1_ready:
                self._add_action_result(new_objects, self.perform_action1())
            if action2_active and action2_ready:
                self._add_action_result(new_objects, self.perform_action2())

        if "action1" in self.released_controls:
            self._add_action_result(new_objects, self.perform_action1_release())

        self.newly_pressed_controls.clear()
        self.released_controls.clear()
        return new_objects

    @staticmethod
    def normalize_action_result(result):
        if not result:
            return []
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

    @classmethod
    def _add_action_result(cls, new_objects, result):
        new_objects.extend(cls.normalize_action_result(result))

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

    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

    def on_ship_impact(self, other, impact):
        """Return optional behavior for a physical collision with another ship."""
        return ShipImpactResult()

    def is_alive(self):
        return self.currently_alive and self.current_hp > 0

    def update(self):
        self.previous_position = self.position.copy()
        self.update_physics()
        return True

    def update_physics(self):
        if self.inertia:
            self.velocity[0] += self.accumulated_impulses[0]
            self.velocity[1] += self.accumulated_impulses[1]
            self.accumulated_impulses = [0.0, 0.0]
            self.apply_verlet()
            self.apply_speed_limit()
        else:
            if self.collision_velocity != [0.0, 0.0]:
                self.velocity = self.collision_velocity.copy()
                self.collision_velocity = [0.0, 0.0]
            else:
                self.velocity = self.accumulated_impulses.copy()
            self.accumulated_impulses = [0.0, 0.0]
            self.apply_speed_limit()
            self.position[0] = (self.position[0] + self.velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE
            self.position[1] = (self.position[1] + self.velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE


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

    def update_timers(self):
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
        if self.energy_timer >= self.energy_wait*const.RECHARGE_DELAY_SCALE:
            self.energy_timer = 0
            if self.current_energy < self.max_energy:
                self.current_energy = min(self.max_energy,
                                          self.current_energy + self.energy_regen)

    def apply_thrust(self, max_thrust, thrust_increment, angle, can_thrust, make_marker):
        if can_thrust:
            angle_rad = math.radians(self.rotation + angle)
            thrust_direction = [math.sin(angle_rad), -math.cos(angle_rad)]

            if self.inertia:
                new_velocity = [
                    self.velocity[0] + thrust_direction[0] * thrust_increment,
                    self.velocity[1] + thrust_direction[1] * thrust_increment
                ]

                speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
                scale = 1.0

                _, planet_distance = self.distance_to(self.planet)
                if speed > max_thrust and planet_distance > const.GRAVITY_RANGE:
                    scale = max_thrust / speed
                if speed > const.MAX_GRAV_WHIP:
                    scale = const.MAX_GRAV_WHIP / speed

                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                diff_vector = [target_velocity[0] - self.velocity[0], target_velocity[1] - self.velocity[1]]

                diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
                if diff_magnitude > thrust_increment:
                    scale = thrust_increment / diff_magnitude
                    self.add_impulse(diff_vector[0] * scale, diff_vector[1] * scale)
                else:
                    self.add_impulse(diff_vector[0] , diff_vector[1] )
            else:
                self.add_impulse(
                    thrust_direction[0] * max_thrust,
                    thrust_direction[1] * max_thrust
                )

            self.thrust_timer = int(self.thrust_wait * const.THRUST_WAIT_SCALE)
            if make_marker:
                marker_x, marker_y = self.get_thrust_marker_position(angle)
                marker = ThrustMarker(marker_x, marker_y)
                return marker
        return None

    def get_thrust_marker_position(self, thrust_angle=0):
        angle_rad = math.radians(self.rotation + thrust_angle)
        offset = (self.size[1] / 2) + 6
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y

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
            self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)

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
            self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)

    def rotation_would_overlap(self):
        try:
            from src.Battle.collisions import ship_rotation_blocked
            return ship_rotation_blocked(self)
        except ImportError:
            return False

    def get_collision_mask(self):
        masks = getattr(self, "masks", None)
        if not masks:
            return None
        return masks[self.heading]

    def execute_action(self, action_number, factory=None):
        """Execute the common action transaction and return its raw result."""
        if not getattr(self, f"can_action{action_number}")():
            return None

        cost = getattr(self, f"a{action_number}_cost")
        wait = getattr(self, f"a{action_number}_wait")
        self.current_energy -= cost
        setattr(
            self,
            f"action{action_number}_timer",
            int(wait * const.ACTION_WAIT_SCALE),
        )

        result = factory(self) if factory else None
        for action_object in self.normalize_action_result(result):
            launch_sound = getattr(action_object, "launch_sound", None)
            if launch_sound:
                launch_sound.play()
                break
        return result

    def perform_action1(self):
        return self.execute_action(1, self.action_factories.get(1))

    def perform_action1_release(self):
        return None

    def perform_action2(self):
        return self.execute_action(2, self.action_factories.get(2))

    def perform_action3(self):
        return None, False

    def set_sprite(self):
        return self.sprites[self.heading]

    def draw(self, screen, scale_factor, translation):
        sprite = self.set_sprite()

        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        # Calculate screen position with translation
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        # Draw the ship at all potential wrap-around positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                # Only draw if the position would be visible
                if (-scaled_rect.width <= pos_x <= const.SCREEN_HEIGHT + scaled_rect.width and
                        -scaled_rect.height <= pos_y <= const.SCREEN_HEIGHT + scaled_rect.height):
                    screen.blit(scaled_sprite, (
                        const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))
