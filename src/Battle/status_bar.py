import math

import pygame
import pygame.gfxdraw

import src.const as const


SPECIAL_INDICATOR_BORDER_COLOR = (0, 0, 0)
SPECIAL_INDICATOR_NEGATIVE_COLOR = (0, 0, 0)


def _draw_indicator_fraction(surface, center, radius, fraction, color):
    """Draw a clockwise pie slice starting at twelve o'clock."""
    fraction = max(0.0, min(1.0, fraction))
    if fraction <= 0.0:
        return
    if fraction >= 1.0:
        pygame.gfxdraw.filled_circle(surface, center, center, radius, color)
        pygame.gfxdraw.aacircle(surface, center, center, radius, color)
        return

    sweep = math.tau * fraction
    point_count = max(2, math.ceil(32 * fraction))
    points = [(center, center)]
    for index in range(point_count + 1):
        angle = -math.pi / 2 + sweep * index / point_count
        points.append(
            (
                round(center + radius * math.cos(angle)),
                round(center + radius * math.sin(angle)),
            )
        )
    pygame.gfxdraw.filled_polygon(surface, points, color)
    pygame.gfxdraw.aapolygon(surface, points, color)


def draw_special_indicator(screen, ship):
    """Draw a ship-provided anti-aliased HUD status light, when present."""
    color = getattr(ship, "hud_indicator_color", None)
    size = getattr(ship, "hud_indicator_size", None)
    gap = getattr(ship, "hud_indicator_gap", None)
    if color is None or size is None or gap is None:
        return

    radius = size // 2
    center = gap + radius
    pygame.gfxdraw.filled_circle(
        screen,
        center,
        center,
        radius,
        SPECIAL_INDICATOR_BORDER_COLOR,
    )
    pygame.gfxdraw.aacircle(
        screen,
        center,
        center,
        radius,
        SPECIAL_INDICATOR_BORDER_COLOR,
    )
    inner_radius = radius - 1
    fraction = getattr(ship, "hud_indicator_fraction", None)
    if fraction is None:
        pygame.gfxdraw.filled_circle(screen, center, center, inner_radius, color)
        pygame.gfxdraw.aacircle(screen, center, center, inner_radius, color)
    else:
        negative_color = getattr(
            ship,
            "hud_indicator_negative_color",
            SPECIAL_INDICATOR_NEGATIVE_COLOR,
        )
        pygame.gfxdraw.filled_circle(
            screen,
            center,
            center,
            inner_radius,
            negative_color,
        )
        pygame.gfxdraw.aacircle(
            screen,
            center,
            center,
            inner_radius,
            negative_color,
        )
        _draw_indicator_fraction(screen, center, inner_radius, fraction, color)


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
        getattr(ship, "crew_bar_color", const.HUD_HP_COLOR),
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
_limpet_counter_font = None

MAX_STATUS_ICON_HEIGHT = 18
LIMPET_COUNTER_FONT_SIZE = 18
LIMPET_COUNTER_GAP = 2
LIMPET_COUNTER_COLOR = (255, 255, 255)


def _get_scaled_icon(icon, new_width, new_height):
    """Return a cached smoothscale result, scaling only on first call per size."""
    key = (id(icon), new_width, new_height)
    result = _scaled_icon_cache.get(key)
    if result is None:
        result = pygame.transform.smoothscale(icon, (new_width, new_height))
        _scaled_icon_cache[key] = result
    return result


def _get_limpet_counter_font():
    global _limpet_counter_font
    if _limpet_counter_font is None:
        _limpet_counter_font = pygame.font.Font(None, LIMPET_COUNTER_FONT_SIZE)
    return _limpet_counter_font


def draw_limpet_count(
    screen, ship, base_x, top_y, total_width, region_height=20
):
    """Draw the active ship form's limpet count centered in a HUD region."""
    count = getattr(ship, "limpets_attached", 0)
    if count <= 0:
        return

    limpet_sprites = ship.resources.ability("VuxA2").sprites
    if not limpet_sprites:
        return

    icon = limpet_sprites[0]
    if icon.get_height() > MAX_STATUS_ICON_HEIGHT:
        scale = MAX_STATUS_ICON_HEIGHT / icon.get_height()
        icon = _get_scaled_icon(
            icon,
            max(1, int(icon.get_width() * scale)),
            MAX_STATUS_ICON_HEIGHT,
        )

    count_text = _get_limpet_counter_font().render(
        f"x{count}", True, LIMPET_COUNTER_COLOR
    )
    content_width = icon.get_width() + LIMPET_COUNTER_GAP + count_text.get_width()
    content_x = base_x + (total_width - content_width) // 2

    screen.blit(
        icon,
        (content_x, top_y + (region_height - icon.get_height()) // 2),
    )
    screen.blit(
        count_text,
        (
            content_x + icon.get_width() + LIMPET_COUNTER_GAP,
            top_y + (region_height - count_text.get_height()) // 2,
        ),
    )


def draw_boarded_marine_icons(
    screen, ship, base_x, top_y, total_width, region_height=20
):
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
    height_scale = 1.0
    raw_max_height = max(icon.get_height() for icon in icons)
    if raw_max_height > MAX_STATUS_ICON_HEIGHT:
        height_scale = MAX_STATUS_ICON_HEIGHT / raw_max_height

    scale = min(width_scale, height_scale)

    scaled_icons = []
    for icon in icons:
        if scale < 1.0:
            new_width = max(1, int(icon.get_width() * scale))
            new_height = max(1, int(icon.get_height() * scale))
            scaled_icons.append(_get_scaled_icon(icon, new_width, new_height))
        else:
            scaled_icons.append(icon)

    total_icons_width = sum(icon.get_width() for icon in scaled_icons) + gap * (
        len(scaled_icons) - 1
    )

    current_x = base_x + (total_width - total_icons_width) // 2
    for icon in scaled_icons:
        icon_y = top_y + (region_height - icon.get_height()) // 2
        screen.blit(icon, (current_x, icon_y))
        current_x += icon.get_width() + gap
