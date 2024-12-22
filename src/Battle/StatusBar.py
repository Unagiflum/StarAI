import pygame
import src.Const as Const


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
    # Colors for HP and Energy
    HP_COLOR = (0, 255, 0)  # Green
    ENERGY_COLOR = (255, 0, 0)  # Red

    # Create HP bar (left)
    hp_bar = StatusBar(base_x, base_y, bar_width, ship.max_hp)
    hp_bar.draw(screen, ship.current_hp, HP_COLOR)

    # Create Energy bar (right)
    energy_bar = StatusBar(base_x + bar_width + bar_spacing, base_y, bar_width, ship.max_energy)
    energy_bar.draw(screen, ship.current_energy, ENERGY_COLOR)