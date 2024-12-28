from src.Objects.object import PlayerObject
import src.const as const
import math
import pygame
import json
from pathlib import Path

# Load projectile data once at module level
with open(const.ABILITIES_JSON_PATH, 'r') as f:
    ABILITIES_DATA = json.load(f)

class Ability(PlayerObject):
    # Class-level storage
    _sprites = {}
    _end_anims = {}
    _launch_sounds = {}

    def __init__(self, ability_name, parent):
        ability_data = ABILITIES_DATA[ability_name]

        # Initialize with temporary size, will be set after sprite loading
        super().__init__(
            name=ability_name,
            sprite_location=Path(ability_data['file_path']),
            sprite_scale=ability_data.get('sprite_scale', 1.0),
            size=[0, 0],
            player=parent.player
        )

        # Load shared resources if not already loaded and sprites are enabled
        if ability_name not in self._sprites:
            self._sprites[ability_name] = []
            self._end_anims[ability_name] = []

            if ability_data.get('has_sprites', True):
                if ability_data['omnidirectional'] and ability_data.get('frames', 1) > 1:
                    # Load animation frames for evolving ability sprites
                    frames = []
                    self.sizes = []
                    for frame in range(ability_data['frames']):
                        frame_path = Path(ability_data['file_path']) / f"{ability_name}00_{frame:02d}.png"
                        base_sprite = pygame.image.load(str(frame_path)).convert_alpha()
                        scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                        frames.append(scaled_sprite)
                        self.sizes.append([scaled_sprite.get_width(), scaled_sprite.get_height()])
                    self._sprites[ability_name].append(frames)
                    self.size = self.sizes[0]
                else:
                    # Load base sprite
                    base_sprite = pygame.image.load(
                        str(Path(ability_data['file_path']) / f"{ability_name}00.png")).convert_alpha()
                    scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                    self._sprites[ability_name].append(scaled_sprite)
                    self.size = [scaled_sprite.get_width(), scaled_sprite.get_height()]

                    # Load additional directional sprites if not omnidirectional
                    if not ability_data['omnidirectional']:
                        for i in range(1, const.SHIP_DIRECTIONS):
                            sprite_path = Path(ability_data['file_path']) / f"{ability_name}{i:02d}.png"
                            base_sprite = pygame.image.load(str(sprite_path)).convert_alpha()
                            scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                            self._sprites[ability_name].append(scaled_sprite)
            else:
                self._sprites[ability_name] = None

            # Load end animation if it exists
            if ability_data.get('end_anim', 0) > 0:
                for i in range(ability_data['end_anim']):
                    try:
                        end_path = Path(ability_data['file_path']) / f"{ability_name}end{i:02d}.png"
                        base_sprite = pygame.image.load(str(end_path)).convert_alpha()
                        scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                        self._end_anims[ability_name].append(scaled_sprite)
                    except pygame.error:
                        break

            # Load sound if it exists and sounds are enabled
            self._launch_sounds[ability_name] = None
            if ability_data.get('has_sound', True):
                try:
                    sound_path = Path(ability_data['file_path']) / f"{ability_name}.wav"
                    self._launch_sounds[ability_name] = pygame.mixer.Sound(str(sound_path))
                except pygame.error:
                    self._launch_sounds[ability_name] = None
        else:
            # Sprites already loaded - set sizes based on existing sprites
            if ability_data.get('has_sprites', True):
                if ability_data['omnidirectional'] and ability_data.get('frames', 1) > 1:
                    # Evolving projectile - get sizes from each frame
                    self.sizes = [[sprite.get_width(), sprite.get_height()] for sprite in self._sprites[ability_name][0]]
                    self.size = self.sizes[0]
                else:
                    # Non-evolving projectile - get size from first sprite
                    self.size = [self._sprites[ability_name][0].get_width(),
                                 self._sprites[ability_name][0].get_height()]
            else:
                self.size = [0, 0]

        self.sprites = self._sprites[ability_name]
        self.death_anim = self._end_anims[ability_name]
        self.launch_sound = self._launch_sounds[ability_name]
        if self.launch_sound:
            self.launch_sound.set_volume(const.SOUND_EFFECT_VOLUME)

        # Rest of initialization code
        self.parent = parent
        self.opponent = self.parent.opponent
        self.planet = self.parent.planet
        self.projectile_name = ability_name

        # Basic properties
        self.type = ability_data['type']
        self.start_hp = ability_data['start_hp'][0]
        self.current_hp = self.start_hp
        self.damages = ability_data['damage']
        self.current_damage = self.damages[0]
        self.tracking = ability_data['tracking']
        self.parent_vel = ability_data['parent_vel']
        self.speed = ability_data['speed'] * const.PROJ_SPEED_SCALE
        self.life_time = ability_data['life_time']
        self.turn_wait = ability_data.get('turn_wait', 0)
        self.inertia = ability_data['inertia']
        self.hit_parent = ability_data['hit_parent']
        self.hit_self = ability_data['hit_self']
        self.omnidirectional = ability_data['omnidirectional']
        self.death_anim = ability_data.get('end_anim', 0)

        # Animation properties
        self.frames = ability_data.get('frames', 1)
        self.frame_delay = ability_data.get('frame_delay', 0)
        self.current_frame = 0
        self.frame_timer = self.frame_delay

        # Store HP array for evolution
        self.hp_array = ability_data['start_hp']

        # State flags
        self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)
        self.can_move = True
        self.can_die = True
        self.can_expire = True

        if self.type == 'laser' or self.type == 'projectile':
            self.can_collide = True
        else:
            self.can_collide = False

        self.expiration_timer = int(self.life_time * const.PROJ_LIFE_SCALE)
        # Load projectile-specific module
        try:
            module_path = f"{ability_data['file_path']}{ability_data['ship_name']}{ability_data['action']}"
            self.projectile_module = __import__(module_path, fromlist=[''])
        except ImportError:
            self.projectile_module = None

    def update_heading(self):

        if self.omnidirectional:
            self.heading = 0
        else:
            direction_step = 360 / const.SHIP_DIRECTIONS
            self.heading = int((self.rotation % 360) / direction_step) % const.SHIP_DIRECTIONS

        if self.tracking and self.opponent:
            # Find opponent
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            # Account for arena wrapping
            if abs(dx) > const.ARENA_SIZE / 2:
                dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
            if abs(dy) > const.ARENA_SIZE / 2:
                dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

            # Calculate target angle
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360

            # Quantize to nearest available direction
            direction_step = const.TURN_ANGLE
            current_angle = self.rotation
            target_direction = round(target_angle / direction_step)
            target_angle = (target_direction * direction_step) % 360

            # Find shortest turning direction
            angle_diff = target_angle - current_angle
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            # Turn if timer allows
            if self.turn_timer <= 0:
                if self.opponent.trackable:
                    if abs(angle_diff) >= direction_step:
                        self.rotation = (current_angle + (direction_step if angle_diff > 0 else -direction_step)) % 360
                        self.turn_timer = int(self.turn_wait * const.TURN_WAIT_SCALE)
                    else:
                        self.rotation = target_angle
            else:
                self.turn_timer -= 1
            angle_rad = math.radians(self.rotation)
            self.velocity = [math.sin(angle_rad) * self.speed, -math.cos(angle_rad) * self.speed]


    def update_physics(self):
        self.update_heading()
        self.apply_speed_limit()
        if self.inertia:
            self.apply_verlet()
        else:
            self.position[0] = (self.position[0] + self.velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE
            self.position[1] = (self.position[1] + self.velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE

    def update(self):
        if not self.currently_alive:
            return False

        self.update_physics()
        self.expiration_timer -= 1

        # Handle frame animation if projectile evolves
        if self.frames > 1:
            if self.frame_timer <= 0:
                if self.current_frame < self.frames - 1:
                    self.current_frame += 1
                    # Update properties for new frame
                    self.size = self.sizes[self.current_frame]
                    self.current_damage = self.damages[self.current_frame]
                    if len(self.hp_array) > 1:
                        self.current_hp = self.hp_array[self.current_frame]
                    self.frame_timer = self.frame_delay
            else:
                self.frame_timer -= 1

        return self.expiration_timer > 0 and self.current_hp > 0

    def on_collide(self, target):
        if not self.hit_parent and target == self.parent:
            return False

        if target.current_hp is not None:
            target.current_hp = max(0, target.current_hp - self.current_damage)

        return True

    def set_hp(self, new_hp):
        """Override hp setting to handle evolution and death"""
        if new_hp <= 0:
            self.current_hp = 0
            return

        if len(self.hp_array) > 1:
            damage_taken = self.current_hp - new_hp
            if damage_taken > 0:
                frame_advance = damage_taken * self.frame_delay
                self.frame_timer -= frame_advance

        self.current_hp = new_hp

    def draw(self, screen, scale_factor, translation):
        if self.frames > 1:
            sprite = self.sprites[0][self.current_frame]  # Get current animation frame
        else:
            sprite = self.sprites[self.heading]

        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (0 <= pos_x <= const.SCREEN_HEIGHT and
                        0 <= pos_y <= const.SCREEN_HEIGHT):
                    screen.blit(scaled_sprite, (
                        const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))