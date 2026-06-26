import pygame

import src.const as const


class StatusBar:
    @staticmethod
    def calculate_height(
        max_value, dash_height=6, dash_gap=2, columns=2, border_thickness=2
    ):
        row_count = (max_value + columns - 1) // columns
        inset = border_thickness + dash_gap
        if row_count == 0:
            return inset * 2
        return inset * 2 + row_count * dash_height + (row_count - 1) * dash_gap

    def __init__(
        self, width, max_value, dash_height=6, dash_gap=2, columns=2, border_thickness=2
    ):
        self.width = width
        self.max_value = max_value
        self.dash_height = dash_height
        self.dash_gap = dash_gap
        self.columns = columns
        self.border_thickness = border_thickness

        self.inset = self.border_thickness + self.dash_gap

        self.dash_width = (
            width - 2 * self.inset - (self.columns - 1) * self.dash_gap
        ) // self.columns
        self.height = self.calculate_height(
            max_value, dash_height, dash_gap, columns, border_thickness
        )

    def draw(self, screen, x, y, current_value, color, border_color, bg_color):
        border_rect = pygame.Rect(x, y, self.width, self.height)
        # Draw background
        pygame.draw.rect(screen, bg_color, border_rect)
        # Draw border
        pygame.draw.rect(screen, border_color, border_rect, self.border_thickness)

        # Calculate dash positions
        dashes_to_draw = current_value
        for i in range(min(self.max_value, dashes_to_draw)):
            column = i % self.columns
            row = i // self.columns

            # Calculate position for each dash
            dash_x = x + self.inset + column * (self.dash_width + self.dash_gap)
            dash_y = (
                (y + self.height)
                - self.inset
                - (row + 1) * self.dash_height
                - row * self.dash_gap
            )

            dash_rect = pygame.Rect(dash_x, dash_y, self.dash_width, self.dash_height)
            pygame.draw.rect(screen, color, dash_rect)


_status_bar_cache = {}


def _get_status_bar(width, max_value):
    """Return a cached StatusBar, creating one if needed."""
    key = (width, max_value)
    bar = _status_bar_cache.get(key)
    if bar is None:
        bar = StatusBar(width, max_value)
        _status_bar_cache[key] = bar
    return bar


def draw_player_status(
    screen, ship, base_x, base_y, bar_width, bar_spacing, viewport_size=0
):

    hp_bar = _get_status_bar(bar_width, ship.max_hp)
    energy_bar = _get_status_bar(bar_width, ship.max_energy)

    # Set positions and align bottoms at base_y
    hp_x = base_x
    energy_x = base_x + bar_width + bar_spacing

    hp_y = base_y - hp_bar.height
    energy_y = base_y - energy_bar.height

    border_color = const.P1_COLOR if ship.player == 1 else const.P2_COLOR

    hp_bar.draw(
        screen,
        hp_x,
        hp_y,
        ship.current_hp,
        const.HUD_HP_COLOR,
        const.HUD_BAR_BORDER,
        const.HUD_BAR_BG,
    )
    energy_bar.draw(
        screen,
        energy_x,
        energy_y,
        ship.current_energy,
        const.HUD_ENERGY_COLOR,
        const.HUD_BAR_BORDER,
        const.HUD_BAR_BG,
    )
    highest_point = hp_y if hp_y < energy_y else energy_y
    if viewport_size > 0:
        viewport_top = base_y - viewport_size
        if viewport_top < highest_point:
            highest_point = viewport_top


_scaled_icon_cache = {}


def _get_scaled_icon(icon, new_width, new_height):
    """Return a cached smoothscale result, scaling only on first call per size."""
    key = (id(icon), new_width, new_height)
    result = _scaled_icon_cache.get(key)
    if result is None:
        result = pygame.transform.smoothscale(icon, (new_width, new_height))
        _scaled_icon_cache[key] = result
    return result


def draw_boarded_marine_icons(screen, ship, base_x, top_y, total_width):
    status_marines = tuple(getattr(ship, "boarded_marines", ()))
    marines = [
        marine
        for marine in status_marines
        if (marine.currently_alive and getattr(marine, "is_boarded", False))
    ]
    if not marines:
        return

    gap = 2
    icons = [marine.hud_sprite for marine in marines]

    raw_width = sum(icon.get_width() for icon in icons) + gap * (len(icons) - 1)

    # Scale based on width constraints
    width_scale = 1.0
    if raw_width > total_width and total_width > 0:
        width_scale = total_width / raw_width

    # Scale based on height constraints (matching bottom padding of roughly 20)
    MAX_ICON_HEIGHT = 18  # 20 - 2 for gap
    height_scale = 1.0
    raw_max_height = max(icon.get_height() for icon in icons)
    if raw_max_height > MAX_ICON_HEIGHT:
        height_scale = MAX_ICON_HEIGHT / raw_max_height

    scale = min(width_scale, height_scale)

    scaled_icons = []
    for icon in icons:
        if scale < 1.0:
            new_width = max(1, int(icon.get_width() * scale))
            new_height = max(1, int(icon.get_height() * scale))
            scaled_icons.append(_get_scaled_icon(icon, new_width, new_height))
        else:
            scaled_icons.append(icon)

    icon_height = max(icon.get_height() for icon in scaled_icons)
    total_icons_width = sum(icon.get_width() for icon in scaled_icons) + gap * (
        len(scaled_icons) - 1
    )

    current_x = base_x + (total_width - total_icons_width) // 2
    for icon in scaled_icons:
        screen.blit(icon, (current_x, top_y))
        current_x += icon.get_width() + gap
