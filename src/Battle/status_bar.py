import pygame

HUD_BORDER = (100, 100, 100)


class StatusBar:
    @staticmethod
    def calculate_height(max_value, dash_height=6, dash_gap=2):
        row_count = (max_value + 1) // 2
        return row_count * (dash_height + dash_gap) + dash_gap

    def __init__(self, x, y, width, max_value, dash_height=6, dash_gap=2):
        self.x = x
        self.y = y
        self.width = width
        self.max_value = max_value
        self.dash_height = dash_height
        self.dash_gap = dash_gap

        self.columns = 2
        self.dash_width = (width - (self.columns + 1) * dash_gap) // self.columns
        self.rows = max_value
        self.height = self.calculate_height(max_value, dash_height, dash_gap)

    def draw(self, screen, current_value, color):
        # Draw border
        border_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, HUD_BORDER, border_rect, 1)

        # Calculate dash positions
        dashes_to_draw = current_value
        for i in range(min(self.rows, dashes_to_draw)):
            column = i % 2  # Alternates between 0 and 1
            row = i // 2

            # Calculate position for each dash
            dash_x = self.x + self.dash_gap + column * (self.dash_width + self.dash_gap)
            dash_y = self.y + self.height - (row + 1) * (self.dash_height + self.dash_gap) + (self.dash_gap // 2)

            dash_rect = pygame.Rect(dash_x, dash_y, self.dash_width, self.dash_height)
            pygame.draw.rect(screen, color, dash_rect)


_status_bar_cache = {}


def _get_status_bar(width, max_value):
    """Return a cached StatusBar, creating one if needed."""
    key = (width, max_value)
    bar = _status_bar_cache.get(key)
    if bar is None:
        bar = StatusBar(0, 0, width, max_value)
        _status_bar_cache[key] = bar
    return bar


def draw_player_status(screen, ship, base_x, base_y, bar_width, bar_spacing,
                       viewport_size=0):
    HP_COLOR = (0, 255, 0)
    ENERGY_COLOR = (255, 0, 0)

    hp_bar = _get_status_bar(bar_width, ship.max_hp)
    energy_bar = _get_status_bar(bar_width, ship.max_energy)

    # Set positions and align bottoms at base_y
    hp_bar.x = base_x
    energy_bar.x = base_x + bar_width + bar_spacing

    hp_y = base_y - hp_bar.height
    energy_y = base_y - energy_bar.height

    hp_bar.y = hp_y
    energy_bar.y = energy_y
    hp_bar.draw(screen, ship.current_hp, HP_COLOR)
    energy_bar.draw(screen, ship.current_energy, ENERGY_COLOR)
    highest_point = hp_y if hp_y < energy_y else energy_y
    if viewport_size > 0:
        viewport_top = base_y - viewport_size
        if viewport_top < highest_point:
            highest_point = viewport_top

    draw_boarded_marine_icons(
        screen,
        ship,
        base_x,
        highest_point,
        bar_width * 2 + bar_spacing,
    )


_scaled_icon_cache = {}


def _get_scaled_icon(icon, new_width, new_height):
    """Return a cached smoothscale result, scaling only on first call per size."""
    key = (id(icon), new_width, new_height)
    result = _scaled_icon_cache.get(key)
    if result is None:
        result = pygame.transform.smoothscale(icon, (new_width, new_height))
        _scaled_icon_cache[key] = result
    return result


def draw_boarded_marine_icons(screen, ship, base_x, highest_point, total_width):
    status_marines = tuple(getattr(ship, "boarded_marines", ()))
    marines = [
        marine for marine in status_marines
        if (
            marine.currently_alive
            and getattr(marine, "is_boarded", False)
        )
    ]
    if not marines:
        return

    gap = 2
    icons = [marine.hud_sprite for marine in marines]

    raw_width = sum(icon.get_width() for icon in icons) + gap * (len(icons) - 1)
    scale = 1.0
    if raw_width > total_width and total_width > 0:
        scale = total_width / raw_width

    scaled_icons = []
    for icon in icons:
        if scale < 1.0:
            new_width = max(1, int(icon.get_width() * scale))
            new_height = max(1, int(icon.get_height() * scale))
            scaled_icons.append(_get_scaled_icon(icon, new_width, new_height))
        else:
            scaled_icons.append(icon)

    icon_height = max(icon.get_height() for icon in scaled_icons)
    total_icons_width = sum(icon.get_width() for icon in scaled_icons) + gap * (len(scaled_icons) - 1)

    top = highest_point - icon_height - gap

    current_x = base_x + (total_width - total_icons_width) // 2
    for icon in scaled_icons:
        screen.blit(icon, (current_x, top))
        current_x += icon.get_width() + gap
