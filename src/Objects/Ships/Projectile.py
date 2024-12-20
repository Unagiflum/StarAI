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
    _death_anims = {}
    _launch_sounds = {}

    def __init__(self, projectile_name, parent):
        projectile_data = PROJECTILES_DATA[projectile_name]

        # Initialize PlayerObject
        super().__init__(
            name=projectile_name,
            sprite_location=Path(projectile_data['Path']),
            sprite_scale=projectile_data.get('SpriteScale', 1.0),
            size=[projectile_data['Size']['width'], projectile_data['Size']['height']],
            player=parent.player
        )
        self.size[0] *= self.sprite_scale
        self.size[1] *= self.sprite_scale
        # Load shared resources if not already loaded
        if projectile_name not in self._sprites:
            self._sprites[projectile_name] = []
            self._death_anims[projectile_name] = []

            # Load base sprite
            base_sprite = pygame.image.load(
                str(Path(projectile_data['Path']) / f"{projectile_name}00.png")).convert_alpha()
            self._sprites[projectile_name].append(base_sprite)

            # Load additional directional sprites only if not omnidirectional
            if not projectile_data['omnidirectional']:
                for i in range(1, Const.SHIP_DIRECTIONS):
                    sprite_path = Path(projectile_data['Path']) / f"{projectile_name}{i:02d}.png"
                    self._sprites[projectile_name].append(pygame.image.load(str(sprite_path)).convert_alpha())

            # Load death animation if it exists
            if projectile_data.get('DeathAnim', 0) > 0:
                for i in range(projectile_data['DeathAnim']):
                    try:
                        death_path = Path(projectile_data['Path']) / f"{projectile_name}die{i:02d}.png"
                        self._death_anims[projectile_name].append(pygame.image.load(str(death_path)).convert_alpha())
                    except pygame.error:
                        break

            # Load sound if it exists
            try:
                sound_path = Path(projectile_data['Path']) / f"{projectile_name}.wav"
                self._launch_sounds[projectile_name] = pygame.mixer.Sound(str(sound_path))
            except pygame.error:
                self._launch_sounds[projectile_name] = None

        self.sprites = self._sprites[projectile_name]
        self.death_anim = self._death_anims[projectile_name]
        self.launch_sound = self._launch_sounds[projectile_name]
        self.launch_sound.set_volume(Const.SOUND_EFFECT_VOLUME)

        # Rest of initialization code
        self.parent = parent
        self.opponent = None

        # Basic properties
        self.start_hp = projectile_data['StartHP']
        self.current_hp = self.start_hp
        self.damage = projectile_data['Damage']
        self.tracking = projectile_data['Tracking']
        self.parent_vel = projectile_data['ParentVel']
        self.speed = projectile_data['Speed'] * Const.PROJ_SPEED_SCALE
        self.life_time = projectile_data['LifeTime']
        self.turn_wait = projectile_data.get('TurnWait', 0)
        self.mass = projectile_data['Mass']
        self.hit_parent = projectile_data['HitParent']
        self.hit_self = projectile_data['HitSelf']
        self.inertia = projectile_data['Inertia']
        self.omnidirectional = projectile_data['omnidirectional']
        self.death_anim = projectile_data.get('DeathAnim', 0)

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

    def update_physics(self):
        # Set sprite index based on whether projectile is omnidirectional
        if self.omnidirectional:
            self.heading = 0
        else:
            # Quantize rotation to nearest available direction
            direction_step = 360 / Const.SHIP_DIRECTIONS
            self.heading = int((self.rotation % 360) / direction_step) % Const.SHIP_DIRECTIONS

        if self.tracking:
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

            # Move in current direction at constant speed
            angle_rad = math.radians(self.rotation)
            self.velocity = [math.sin(angle_rad) * self.speed, -math.cos(angle_rad) * self.speed]

        # Update position based on velocity
        self.position[0] = (self.position[0] + self.velocity[0] * Const.SPEED_SCALE) % Const.ARENA_SIZE
        self.position[1] = (self.position[1] + self.velocity[1] * Const.SPEED_SCALE) % Const.ARENA_SIZE

    def update(self):
        if not self.currently_alive:
            return False

        self.update_physics()
        self.expiration_timer -= 1
        return self.expiration_timer > 0

    def on_collide(self, target):
        if not self.hit_parent and target == self.parent:
            return False

        if target.current_hp is not None:
            target.current_hp = max(0, target.current_hp - self.damage)

        return True

    def draw(self, screen, scale_factor, translation):
        sprite = self.sprites[self.heading]
        total_scale = scale_factor * self.sprite_scale
        scaled_sprite = pygame.transform.smoothscale_by(sprite, total_scale)
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