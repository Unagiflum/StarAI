from src.Objects.Object import PlayerObject
import src.Const as Const
import math
import pygame
import json
from pathlib import Path

# Load projectile data once at module level
with open(Const.PROJECTILES_JSON_PATH, 'r') as f:
    PROJECTILES_DATA = json.load(f)

class Projectile(PlayerObject):
    # Class-level storage
    _sprites = {}
    _end_anims = {}
    _launch_sounds = {}

    def __init__(self, projectile_name, parent):
        projectile_data = PROJECTILES_DATA[projectile_name]

        # Initialize with temporary size, will be set after sprite loading
        super().__init__(
            name=projectile_name,
            sprite_location=Path(projectile_data['Path']),
            sprite_scale=projectile_data.get('SpriteScale', 1.0),
            size=[0, 0],
            player=parent.player
        )

        # Load shared resources if not already loaded and sprites are enabled
        if projectile_name not in self._sprites:
            self._sprites[projectile_name] = []
            self._end_anims[projectile_name] = []

            if projectile_data.get('hasSprites', True):
                if projectile_data['omnidirectional'] and projectile_data.get('frames', 1) > 1:
                    # Load animation frames for evolving projectile
                    frames = []
                    self.sizes = []
                    for frame in range(projectile_data['frames']):
                        frame_path = Path(projectile_data['Path']) / f"{projectile_name}00_{frame:02d}.png"
                        base_sprite = pygame.image.load(str(frame_path)).convert_alpha()
                        scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                        frames.append(scaled_sprite)
                        self.sizes.append([scaled_sprite.get_width(), scaled_sprite.get_height()])
                    self._sprites[projectile_name].append(frames)
                    self.size = self.sizes[0]
                else:
                    # Load base sprite
                    base_sprite = pygame.image.load(
                        str(Path(projectile_data['Path']) / f"{projectile_name}00.png")).convert_alpha()
                    scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                    self._sprites[projectile_name].append(scaled_sprite)
                    self.size = [scaled_sprite.get_width(), scaled_sprite.get_height()]

                    # Load additional directional sprites if not omnidirectional
                    if not projectile_data['omnidirectional']:
                        for i in range(1, Const.SHIP_DIRECTIONS):
                            sprite_path = Path(projectile_data['Path']) / f"{projectile_name}{i:02d}.png"
                            base_sprite = pygame.image.load(str(sprite_path)).convert_alpha()
                            scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                            self._sprites[projectile_name].append(scaled_sprite)
            else:
                self._sprites[projectile_name] = None

            # Load end animation if it exists
            if projectile_data.get('end_anim', 0) > 0:
                for i in range(projectile_data['end_anim']):
                    try:
                        end_path = Path(projectile_data['Path']) / f"{projectile_name}end{i:02d}.png"
                        base_sprite = pygame.image.load(str(end_path)).convert_alpha()
                        scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                        self._end_anims[projectile_name].append(scaled_sprite)
                    except pygame.error:
                        break

            # Load sound if it exists and sounds are enabled
            if projectile_data.get('hasSound', True):
                try:
                    sound_path = Path(projectile_data['Path']) / f"{projectile_name}.wav"
                    self._launch_sounds[projectile_name] = pygame.mixer.Sound(str(sound_path))
                except pygame.error:
                    self._launch_sounds[projectile_name] = None
        else:
            # Sprites already loaded - set sizes based on existing sprites
            if projectile_data.get('hasSprites', True):
                if projectile_data['omnidirectional'] and projectile_data.get('frames', 1) > 1:
                    # Evolving projectile - get sizes from each frame
                    self.sizes = [[sprite.get_width(), sprite.get_height()] for sprite in self._sprites[projectile_name][0]]
                    self.size = self.sizes[0]
                else:
                    # Non-evolving projectile - get size from first sprite
                    self.size = [self._sprites[projectile_name][0].get_width(),
                                 self._sprites[projectile_name][0].get_height()]
            else:
                self.size = [0, 0]

        self.sprites = self._sprites[projectile_name]
        self.death_anim = self._end_anims[projectile_name]
        self.launch_sound = self._launch_sounds[projectile_name]
        if self.launch_sound:
            self.launch_sound.set_volume(Const.SOUND_EFFECT_VOLUME)

        # Rest of initialization code
        self.parent = parent
        self.opponent = self.parent.opponent
        self.planet = self.parent.planet
        self.projectile_name = projectile_name

        # Basic properties
        self.start_hp = projectile_data['StartHP'][0]
        self.current_hp = self.start_hp
        self.damages = projectile_data['Damage']
        self.current_damage = self.damages[0]
        self.tracking = projectile_data['Tracking']
        self.parent_vel = projectile_data['ParentVel']
        self.speed = projectile_data['Speed'] * Const.PROJ_SPEED_SCALE
        self.life_time = projectile_data['LifeTime']
        self.turn_wait = projectile_data.get('TurnWait', 0)
        self.inertia = projectile_data['Inertia']
        self.hit_parent = projectile_data['HitParent']
        self.hit_self = projectile_data['HitSelf']
        self.omnidirectional = projectile_data['omnidirectional']
        self.death_anim = projectile_data.get('end_anim', 0)

        # Animation properties
        self.frames = projectile_data.get('frames', 1)
        self.frame_delay = projectile_data.get('frame_delay', 0)
        self.current_frame = 0
        self.frame_timer = self.frame_delay

        # Store HP array for evolution
        self.hp_array = projectile_data['StartHP']

        # State flags
        self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)
        self.can_move = True
        self.can_die = True
        self.can_expire = True
        self.expiration_timer = int(self.life_time*Const.PROJ_LIFE_SCALE)
        # Load projectile-specific module
        try:
            module_path = f"{projectile_data['Path']}{projectile_data['ShipName']}{projectile_data['Action']}"
            self.projectile_module = __import__(module_path, fromlist=[''])
        except ImportError:
            self.projectile_module = None

    def update_heading(self):

        if self.omnidirectional:
            self.heading = 0
        else:
            direction_step = 360 / Const.SHIP_DIRECTIONS
            self.heading = int((self.rotation % 360) / direction_step) % Const.SHIP_DIRECTIONS

        if self.tracking and self.opponent:
            # Find opponent
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            # Account for arena wrapping
            if abs(dx) > Const.ARENA_SIZE / 2:
                dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
            if abs(dy) > Const.ARENA_SIZE / 2:
                dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE

            # Calculate target angle
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360

            # Quantize to nearest available direction
            direction_step = 360 / Const.SHIP_DIRECTIONS
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
                if abs(angle_diff) >= direction_step:
                    self.rotation = (current_angle + (direction_step if angle_diff > 0 else -direction_step)) % 360
                    self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)
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
            self.position[0] = (self.position[0] + self.velocity[0] * Const.SPEED_SCALE) % Const.ARENA_SIZE
            self.position[1] = (self.position[1] + self.velocity[1] * Const.SPEED_SCALE) % Const.ARENA_SIZE

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
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (0 <= pos_x <= Const.SCREEN_HEIGHT and
                        0 <= pos_y <= Const.SCREEN_HEIGHT):
                    screen.blit(scaled_sprite, (
                        Const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))