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
        return (self.parent.heading + self.relative_heading) % const.SHIP_DIRECTIONS

    def reset(self):
        self.relative_heading = 0

    def turn(self, direction):
        self.relative_heading = (
            self.relative_heading + direction
        ) % const.SHIP_DIRECTIONS

    @property
    def previous_absolute_heading(self):
        prev_parent = getattr(self.parent, "previous_heading", self.parent.heading)
        prev_rel = getattr(self, "previous_relative_heading", self.relative_heading)
        return (prev_parent + prev_rel) % const.SHIP_DIRECTIONS

    def get_sprite(self, interp_t=0.0):
        if const.VIDEO_FPS_MULTIPLIER <= 1:
            return self.sprites[const.heading_to_sprite_index(self.absolute_heading)]

        prev_heading = self.previous_absolute_heading
        curr_heading = self.absolute_heading

        diff = curr_heading - prev_heading
        if diff > const.SHIP_DIRECTIONS // 2:
            diff -= const.SHIP_DIRECTIONS
        elif diff < -(const.SHIP_DIRECTIONS // 2):
            diff += const.SHIP_DIRECTIONS

        prev_sprite = prev_heading * const.VIDEO_FPS_MULTIPLIER
        sprite_diff = diff * const.VIDEO_FPS_MULTIPLIER
        interp_sprite = prev_sprite + sprite_diff * interp_t
        idx = round(interp_sprite) % const.TOTAL_SPRITE_DIRECTIONS
        return self.sprites[idx]
