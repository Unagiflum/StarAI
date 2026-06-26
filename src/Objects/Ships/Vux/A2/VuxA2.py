from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import math
import src.const as const


class VuxA2(Ability):
    def __init__(self, parent):
        super().__init__("VuxA2", parent)
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
        angle_rad = math.radians(self.parent.rotation)
        spawn_distance = (self.parent.size[1]) / 2
        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]

        opponent = getattr(self.parent, "opponent", None)
        if opponent:
            from src.toroidal import wrapped_delta

            dx, dy = wrapped_delta(self.position, opponent.position)
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360
            direction_step = 360 / const.SHIP_DIRECTIONS
            self.heading = round(target_angle / direction_step) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE
        else:
            self.heading = self.parent.heading
            self.rotation = self.parent.rotation

    def handle_ship_contact(self, ship, normal=None):
        if hasattr(ship, "attach_limpet"):
            ship.attach_limpet()
        if getattr(self, "bite_sound", None):
            self.bite_sound.play()

        self.current_hp = 0
        self.currently_alive = False
        return True
