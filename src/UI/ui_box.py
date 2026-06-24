import pygame
from . import ui
import src.const as Const
from src.menu_state import FleetModel

# Constants for ShipList and Fleet
SCROLLBAR_WIDTH = int(0.01*Const.SCREEN_WIDTH)
SCROLL_BUTTON_HEIGHT = int(0.01*Const.SCREEN_HEIGHT)
SCROLL_SPEED = 20
SPACING = int(0.005*Const.SCREEN_WIDTH)


def create_player_fleet_panels(column_lefts, top, width, height, icon_size):
    """Create the two existing player fleet panels from shared layout data."""
    return {
        player: Fleet(
            column_lefts[player], top, width, height,
            f"Player {player} Fleet", icon_size,
            color=Const.P1_COLOR if player == 1 else Const.P2_COLOR
        )
        for player in (1, 2)
    }

class ShipContainer:
    def __init__(self, x, y, width, height, title, icon_size, color=ui.WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.icon_size = icon_size
        self.color = color
        self.ships = []
        self.spacing = SPACING
        self.icons_per_row = 1
        self.calculate_tiling()

    def calculate_tiling(self):
        self.available_width = self.rect.width - 2 * SPACING
        self.icons_per_row = self.available_width // (self.icon_size[0] + self.spacing)
        if self.icons_per_row < 1:
            self.icons_per_row = 1
        total_icons_width = self.icons_per_row * self.icon_size[0]
        if self.icons_per_row > 1:
            self.spacing = (self.available_width - total_icons_width) // (self.icons_per_row - 1)
        else:
            self.spacing = 0
        self.icon_total_height = self.icon_size[1] + SPACING + 5
        self.max_fleet_size = self.icons_per_row * ((self.rect.height - 40) // self.icon_total_height)

    def add_ship(self, sprite, name, cost):
        """To be implemented by subclasses."""
        pass

    def handle_event(self, event):
        """To be implemented by subclasses."""
        pass

    def draw(self, screen, font, player_font=None):
        """To be implemented by subclasses."""
        pass

class ShipList(ShipContainer):
    def __init__(self, x, y, width, height, title, icon_size, color=ui.WHITE):
        super().__init__(x, y, width, height, title, icon_size, color=color)
        self.scroll_y = 0
        self.dragging = False
        self.spacing = SPACING
        self.calculate_tiling()
        self.create_scrollbar()

    def calculate_tiling(self):
        self.available_width = self.rect.width - 3 * SPACING - SCROLLBAR_WIDTH
        self.icons_per_row = self.available_width // (self.icon_size[0] + self.spacing)
        if self.icons_per_row < 1:
            self.icons_per_row = 1
        total_icons_width = self.icons_per_row * self.icon_size[0]
        self.spacing = (self.available_width - total_icons_width) // (self.icons_per_row + 1)
        self.icon_total_height = self.icon_size[1] + SPACING + 24 + 5  # 24 for cost text height
        self.visible_rows = (self.rect.height - 40) // self.icon_total_height
        total_rows = (len(self.ships) + self.icons_per_row - 1) // self.icons_per_row
        self.max_scroll = max(0, (total_rows - self.visible_rows) * self.icon_total_height)

    def create_scrollbar(self):
        self.scrollbar_rect = pygame.Rect(
            self.rect.right - SCROLLBAR_WIDTH - SPACING,
            self.rect.y + SPACING,
            SCROLLBAR_WIDTH,
            self.rect.height - 2 * SPACING
        )
        self.scroll_up_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scrollbar_rect.y,
            SCROLLBAR_WIDTH,
            SCROLL_BUTTON_HEIGHT
        )
        self.scroll_down_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scrollbar_rect.bottom - SCROLL_BUTTON_HEIGHT,
            SCROLLBAR_WIDTH,
            SCROLL_BUTTON_HEIGHT
        )
        self.scroll_track_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scroll_up_rect.bottom,
            SCROLLBAR_WIDTH,
            self.scrollbar_rect.height - 2 * SCROLL_BUTTON_HEIGHT
        )
        self.update_scrollbar_thumb()

    def update_scrollbar_thumb(self):
        if self.max_scroll == 0:
            thumb_height = self.scroll_track_rect.height
        else:
            thumb_height = max(30, self.scroll_track_rect.height * self.visible_rows // (
                        (len(self.ships) + self.icons_per_row - 1) // self.icons_per_row))

        scroll_ratio = self.scroll_y / self.max_scroll if self.max_scroll > 0 else 0
        thumb_y = self.scroll_track_rect.y + scroll_ratio * (self.scroll_track_rect.height - thumb_height)

        self.scroll_thumb_rect = pygame.Rect(
            self.scroll_track_rect.x,
            thumb_y,
            SCROLLBAR_WIDTH,
            thumb_height
        )

    def add_ship(self, sprite, name, cost):
        ship_info = (sprite, name, cost, None)
        self.ships.append(ship_info)
        self.calculate_tiling()
        self.update_scrollbar_thumb()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if event.button == 4:
                if self.rect.collidepoint(mouse_pos):
                    self.scroll(-SCROLL_SPEED)
            elif event.button == 5:
                if self.rect.collidepoint(mouse_pos):
                    self.scroll(SCROLL_SPEED)
            elif event.button == 1:
                if self.scroll_up_rect.collidepoint(mouse_pos):
                    self.scroll(-SCROLL_SPEED)
                elif self.scroll_down_rect.collidepoint(mouse_pos):
                    self.scroll(SCROLL_SPEED)
                elif self.scroll_thumb_rect.collidepoint(mouse_pos):
                    self.dragging = True
                    self.drag_offset = mouse_pos[1] - self.scroll_thumb_rect.y
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_y = event.pos[1]
                new_thumb_y = mouse_y - self.drag_offset
                new_thumb_y = max(self.scroll_track_rect.y,
                                  min(new_thumb_y,
                                      self.scroll_track_rect.bottom - self.scroll_thumb_rect.height))
                scroll_ratio = (new_thumb_y - self.scroll_track_rect.y) / (
                            self.scroll_track_rect.height - self.scroll_thumb_rect.height)
                self.scroll_y = scroll_ratio * self.max_scroll
                self.update_scrollbar_thumb()

    def scroll(self, amount):
        self.scroll_y = max(0, min(self.scroll_y + amount, self.max_scroll))
        self.update_scrollbar_thumb()

    def draw_scrollbar(self, screen):
        pygame.draw.rect(screen, ui.GREY, self.scroll_up_rect)
        pygame.draw.rect(screen, ui.GREY, self.scroll_down_rect)
        font = pygame.font.SysFont(None, 20)
        up_arrow = font.render("^", True, ui.BLACK)
        down_arrow = font.render("v", True, ui.BLACK)
        up_rect = up_arrow.get_rect(center=self.scroll_up_rect.center)
        down_rect = down_arrow.get_rect(center=self.scroll_down_rect.center)
        screen.blit(up_arrow, up_rect)
        screen.blit(down_arrow, down_rect)
        pygame.draw.rect(screen, ui.DARK_GREY, self.scroll_track_rect)
        pygame.draw.rect(screen, ui.WHITE, self.scroll_thumb_rect)

    def draw(self, screen, font, player_font):
        pygame.draw.rect(screen, ui.BLACK, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)

        title_text = player_font.render(self.title, True, ui.WHITE)
        title_rect = title_text.get_rect(x=self.rect.x + 10, y=self.rect.y + 10)
        screen.blit(title_text, title_rect)

        clip_rect = pygame.Rect(
            self.rect.x + SPACING,
            self.rect.y + 40,
            self.available_width,
            self.rect.height - 40 - SPACING
        )
        screen.set_clip(clip_rect)

        for index, (sprite, name, cost, _) in enumerate(self.ships):
            row = index // self.icons_per_row
            col = index % self.icons_per_row

            slot_x = self.rect.x + SPACING + col * (self.icon_size[0] + self.spacing)
            slot_y = self.rect.y + 40 + row * self.icon_total_height - self.scroll_y

            ship_rect = pygame.Rect(slot_x, slot_y, self.icon_size[0], self.icon_size[1])

            sprite_rect = sprite.get_rect(center=ship_rect.center)
            self.ships[index] = (sprite, name, cost, sprite_rect)

            if ship_rect.colliderect(clip_rect):
                screen.blit(sprite, sprite_rect)
                cost_text = font.render(str(cost), True, ui.WHITE)
                cost_rect = cost_text.get_rect(centerx=ship_rect.centerx, top=ship_rect.bottom + 2)
                screen.blit(cost_text, cost_rect)

        screen.set_clip(None)
        self.draw_scrollbar(screen)

class Fleet(ShipContainer):
    """Pygame presentation for an ordered :class:`FleetModel`."""

    def __init__(self, x, y, width, height, title, icon_size, model=None, color=ui.WHITE):
        super().__init__(x, y, width, height, title, icon_size, color=color)
        self.calculate_tiling()
        self.model = model if model is not None else FleetModel(self.max_fleet_size)
        if self.model.capacity != self.max_fleet_size:
            raise ValueError("Fleet model capacity must match the panel layout")

    def calculate_tiling(self):
        # Subtract margins from available space
        available_width = self.rect.width - (2 * SPACING)
        available_height = self.rect.height - 40 - (3 * SPACING)  # 40 for title, extra SPACING for bottom margin

        # Calculate maximum space we can use for each icon including spacing
        total_vertical_space = available_height // Const.SHIP_ROWS
        total_horizontal_space = available_width // Const.SHIP_COLS

        # Subtract spacing to get actual icon size, ensuring square icons
        icon_size = min(
            total_horizontal_space - SPACING,
            total_vertical_space - SPACING
        )

        self.icon_size = (icon_size, icon_size)

        # Calculate grid dimensions
        grid_width = (Const.SHIP_COLS * icon_size) + ((Const.SHIP_COLS - 1) * SPACING)

        # Center the grid horizontally
        self.left_offset = (available_width - grid_width) // 2 + SPACING

        self.spacing = SPACING
        self.icons_per_row = Const.SHIP_COLS
        self.icon_total_height = self.icon_size[1] + SPACING
        self.max_fleet_size = Const.SHIP_COLS * Const.SHIP_ROWS

    def add_ship(self, sprite, name, cost):
        if self.model.add_ship(name, cost):
            row = len(self.ships) // self.icons_per_row
            col = len(self.ships) % self.icons_per_row

            slot_x = self.rect.x + self.left_offset + col * (self.icon_size[0] + self.spacing)
            slot_y = self.rect.y + 40 + row * self.icon_total_height

            ship_rect = pygame.Rect(slot_x, slot_y, self.icon_size[0], self.icon_size[1])
            sprite_rect = sprite.get_rect(center=ship_rect.center)

            self.ships.append((sprite, name, cost, sprite_rect))
            return True
        return False

    def clear(self):
        self.model.clear()
        self.ships.clear()

    def remove_ship_at_pos(self, pos):
        for i, (_, _, _, rect) in enumerate(self.ships):
            if rect.collidepoint(pos):
                self.ships.pop(i)
                self.model.remove_ship(i)
                self._update_ship_positions()
                return True
        return False

    def _update_ship_positions(self):
        for i, (sprite, name, cost, _) in enumerate(self.ships):
            row = i // self.icons_per_row
            col = i % self.icons_per_row

            slot_x = self.rect.x + self.left_offset + col * (self.icon_size[0] + self.spacing)
            slot_y = self.rect.y + 40 + row * self.icon_total_height

            slot_rect = pygame.Rect(slot_x, slot_y, self.icon_size[0], self.icon_size[1])
            sprite_rect = sprite.get_rect(center=slot_rect.center)

            self.ships[i] = (sprite, name, cost, sprite_rect)

    def get_total_cost(self):
        return self.model.total_cost

    def draw(self, screen, font, player_font=None):
        pygame.draw.rect(screen, ui.BLACK, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)

        title_text = font.render(f"{self.title} - cost: {self.get_total_cost()}", True, ui.WHITE)
        screen.blit(title_text, (self.rect.x + 10, self.rect.y + 10))

        for sprite, _, _, rect in self.ships:
            screen.blit(sprite, rect)
