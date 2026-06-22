import src.const as const


class OrzA2:
    """Persistent visual and direction state for the Orz turret."""

    def __init__(self, parent):
        self.name = "OrzA2"
        self.relative_heading = 0
        self.parent = parent
        self.resources = parent.resources
        self.sprites = self.resources.ability(self.name).sprites

    @property
    def absolute_heading(self):
        return (
            self.parent.heading + self.relative_heading
        ) % const.SHIP_DIRECTIONS

    def reset(self):
        self.relative_heading = 0

    def turn(self, direction):
        self.relative_heading = (
            self.relative_heading + direction
        ) % const.SHIP_DIRECTIONS

    def get_sprite(self):
        return self.sprites[self.absolute_heading]
