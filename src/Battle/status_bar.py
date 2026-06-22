import pygame
import src.const as const


class StatusBar:
    def __init__(self, x, y, width, max_value, dash_height=6, dash_gap=2):  # Increased dash_height
        self.x = x
        self.y = y
        self.width = width
        self.max_value = max_value
        self.dash_height = dash_height
        self.dash_gap = dash_gap

        self.columns = 2
        self.dash_width = (width - (self.columns + 1) * dash_gap) // self.columns
        self.rows = max_value
        self.height = ((self.rows + 1) // 2) * (dash_height + dash_gap) + dash_gap

    def draw(self, screen, current_value, color):
        # Draw border
        border_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, (100, 100, 100), border_rect, 1)

        # Calculate dash positions
        dashes_to_draw = current_value
        for i in range(min(self.rows, dashes_to_draw)):
            column = i % 2  # Alternates between 0 and 1
            row = i // 2

            # Calculate position for each dash
            dash_x = self.x + self.dash_gap + column * (self.dash_width + self.dash_gap)
            dash_y = self.y + self.height - (row + 1) * (self.dash_height + self.dash_gap)

            dash_rect = pygame.Rect(dash_x, dash_y, self.dash_width, self.dash_height)
            pygame.draw.rect(screen, color, dash_rect)


def draw_player_status(screen, ship, base_x, base_y, bar_width, bar_spacing):
    HP_COLOR = (0, 255, 0)
    ENERGY_COLOR = (255, 0, 0)

    # Create bars to calculate heights
    hp_bar = StatusBar(base_x, 0, bar_width, ship.max_hp)
    energy_bar = StatusBar(base_x + bar_width + bar_spacing, 0, bar_width, ship.max_energy)

    # Calculate y positions to align bottoms at base_y
    hp_y = base_y - hp_bar.height
    energy_y = base_y - energy_bar.height

    # Update bar positions and draw
    hp_bar.y = hp_y
    energy_bar.y = energy_y
    hp_bar.draw(screen, ship.current_hp, HP_COLOR)
    energy_bar.draw(screen, ship.current_energy, ENERGY_COLOR)
    draw_boarded_marine_icons(
        screen,
        ship,
        base_x,
        min(hp_y, energy_y),
        bar_width * 2 + bar_spacing,
    )


def draw_boarded_marine_icons(screen, ship, base_x, bars_top, total_width):
    marines = [
        marine for marine in getattr(ship, "boarded_marines", ())
        if marine.currently_alive and getattr(marine, "is_boarded", False)
    ]
    if not marines:
        return

    icon_size = 12
    gap = 2
    columns = max(1, total_width // (icon_size + gap))
    rows = (len(marines) + columns - 1) // columns
    top = bars_top - rows * (icon_size + gap)
    for index, marine in enumerate(marines):
        row, column = divmod(index, columns)
        row_count = min(columns, len(marines) - row * columns)
        row_width = row_count * icon_size + (row_count - 1) * gap
        x = base_x + (total_width - row_width) // 2 + column * (icon_size + gap)
        y = top + row * (icon_size + gap)
        icon = marine.hud_sprite
        screen.blit(icon, (x, y))
