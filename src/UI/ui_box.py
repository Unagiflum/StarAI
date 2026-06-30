import pygame
from . import ui
import src.const as Const
from src.menu_state import FleetModel

# Constants for ShipList and Fleet
SCROLLBAR_WIDTH = int(0.01 * Const.SCREEN_WIDTH)
SCROLL_BUTTON_HEIGHT = int(0.01 * Const.SCREEN_HEIGHT)
SCROLL_SPEED = 20
SPACING = int(0.005 * Const.SCREEN_WIDTH)
FLEET_SLOT_SPACING = 3
FLEET_SLOT_COLOR = Const.SHIP_PANEL_BACKGROUND_COLOR
FLEET_BOX_COLOR = Const.SHIP_BOX_BACKGROUND_COLOR
FLEET_BORDER_WIDTH = 3
FLEET_EDGE_LINE_WIDTH = 3
FLEET_CONTENT_INSET = FLEET_BORDER_WIDTH + FLEET_EDGE_LINE_WIDTH
FLEET_TITLE_HEIGHT = 40
SHIP_SELECTION_HOVER_FADE_MS = 1200


def create_player_fleet_panels(
    column_lefts,
    top,
    width,
    height,
    icon_size,
    fleet_factory=None,
):
    """Create the two existing player fleet panels from shared layout data."""
    fleet_factory = fleet_factory or Fleet
    return {
        player: fleet_factory(
            column_lefts[player],
            top,
            width,
            height,
            f"Player {player} Fleet",
            icon_size,
            color=Const.P1_COLOR if player == 1 else Const.P2_COLOR,
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
            self.spacing = (self.available_width - total_icons_width) // (
                self.icons_per_row - 1
            )
        else:
            self.spacing = 0
        self.icon_total_height = self.icon_size[1] + SPACING + 5
        self.max_fleet_size = self.icons_per_row * (
            (self.rect.height - 40) // self.icon_total_height
        )

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
        self.spacing = (self.available_width - total_icons_width) // (
            self.icons_per_row + 1
        )
        self.icon_total_height = (
            self.icon_size[1] + SPACING + 24 + 5
        )  # 24 for cost text height
        self.visible_rows = (self.rect.height - 40) // self.icon_total_height
        total_rows = (len(self.ships) + self.icons_per_row - 1) // self.icons_per_row
        self.max_scroll = max(
            0, (total_rows - self.visible_rows) * self.icon_total_height
        )

    def create_scrollbar(self):
        self.scrollbar_rect = pygame.Rect(
            self.rect.right - SCROLLBAR_WIDTH - SPACING,
            self.rect.y + SPACING,
            SCROLLBAR_WIDTH,
            self.rect.height - 2 * SPACING,
        )
        self.scroll_up_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scrollbar_rect.y,
            SCROLLBAR_WIDTH,
            SCROLL_BUTTON_HEIGHT,
        )
        self.scroll_down_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scrollbar_rect.bottom - SCROLL_BUTTON_HEIGHT,
            SCROLLBAR_WIDTH,
            SCROLL_BUTTON_HEIGHT,
        )
        self.scroll_track_rect = pygame.Rect(
            self.scrollbar_rect.x,
            self.scroll_up_rect.bottom,
            SCROLLBAR_WIDTH,
            self.scrollbar_rect.height - 2 * SCROLL_BUTTON_HEIGHT,
        )
        self.update_scrollbar_thumb()

    def update_scrollbar_thumb(self):
        if self.max_scroll == 0:
            thumb_height = self.scroll_track_rect.height
        else:
            thumb_height = max(
                30,
                self.scroll_track_rect.height
                * self.visible_rows
                // ((len(self.ships) + self.icons_per_row - 1) // self.icons_per_row),
            )

        scroll_ratio = self.scroll_y / self.max_scroll if self.max_scroll > 0 else 0
        thumb_y = self.scroll_track_rect.y + scroll_ratio * (
            self.scroll_track_rect.height - thumb_height
        )

        self.scroll_thumb_rect = pygame.Rect(
            self.scroll_track_rect.x, thumb_y, SCROLLBAR_WIDTH, thumb_height
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
                new_thumb_y = max(
                    self.scroll_track_rect.y,
                    min(
                        new_thumb_y,
                        self.scroll_track_rect.bottom - self.scroll_thumb_rect.height,
                    ),
                )
                scroll_ratio = (new_thumb_y - self.scroll_track_rect.y) / (
                    self.scroll_track_rect.height - self.scroll_thumb_rect.height
                )
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
        pygame.draw.rect(screen, self.color, self.rect, 3)

        title_text = player_font.render(self.title, True, ui.WHITE)
        title_rect = title_text.get_rect(x=self.rect.x + 10, y=self.rect.y + 10)
        screen.blit(title_text, title_rect)

        clip_rect = pygame.Rect(
            self.rect.x + SPACING,
            self.rect.y + 40,
            self.available_width,
            self.rect.height - 40 - SPACING,
        )
        screen.set_clip(clip_rect)

        for index, (sprite, name, cost, _) in enumerate(self.ships):
            row = index // self.icons_per_row
            col = index % self.icons_per_row

            slot_x = self.rect.x + SPACING + col * (self.icon_size[0] + self.spacing)
            slot_y = self.rect.y + 40 + row * self.icon_total_height - self.scroll_y

            ship_rect = pygame.Rect(
                slot_x, slot_y, self.icon_size[0], self.icon_size[1]
            )

            sprite_rect = sprite.get_rect(center=ship_rect.center)
            self.ships[index] = (sprite, name, cost, sprite_rect)

            if ship_rect.colliderect(clip_rect):
                screen.blit(sprite, sprite_rect)
                cost_text = font.render(str(cost), True, ui.WHITE)
                cost_rect = cost_text.get_rect(
                    centerx=ship_rect.centerx, top=ship_rect.bottom + 2
                )
                screen.blit(cost_text, cost_rect)

        screen.set_clip(None)
        self.draw_scrollbar(screen)


class Fleet(ShipContainer):
    """Pygame presentation for an ordered :class:`FleetModel`."""

    def __init__(
        self, x, y, width, height, title, icon_size, model=None, color=ui.WHITE
    ):
        super().__init__(x, y, width, height, title, icon_size, color=color)
        self.calculate_tiling()
        self.model = model if model is not None else FleetModel(self.max_fleet_size)
        if self.model.capacity != self.max_fleet_size:
            raise ValueError("Fleet model capacity must match the panel layout")
        self.ships = [None] * self.max_fleet_size

    def calculate_tiling(self):
        available_width = self.rect.width - 2 * FLEET_CONTENT_INSET
        available_height = (
            self.rect.height - FLEET_TITLE_HEIGHT - FLEET_CONTENT_INSET
        )
        horizontal_gaps = (Const.SHIP_COLS - 1) * FLEET_SLOT_SPACING
        vertical_gaps = (Const.SHIP_ROWS - 1) * FLEET_SLOT_SPACING
        icon_size = max(
            1,
            min(
                (available_width - horizontal_gaps) // Const.SHIP_COLS,
                (available_height - vertical_gaps) // Const.SHIP_ROWS,
            ),
        )

        self.icon_size = (icon_size, icon_size)

        grid_width = (Const.SHIP_COLS * icon_size) + (
            (Const.SHIP_COLS - 1) * FLEET_SLOT_SPACING
        )
        grid_height = (Const.SHIP_ROWS * icon_size) + vertical_gaps

        self.left_offset = FLEET_CONTENT_INSET + (available_width - grid_width) // 2
        self.top_offset = FLEET_TITLE_HEIGHT + (
            available_height - grid_height
        ) // 2

        self.spacing = FLEET_SLOT_SPACING
        self.icons_per_row = Const.SHIP_COLS
        self.icon_total_height = self.icon_size[1] + FLEET_SLOT_SPACING
        self.max_fleet_size = Const.SHIP_COLS * Const.SHIP_ROWS

    def add_ship(self, sprite, name, cost):
        for index in range(self.max_fleet_size):
            if self.ships[index] is None:
                return self.set_ship_at_slot(index, sprite, name, cost)
        return False

    def set_ship_at_slot(self, index, sprite, name, cost):
        """Replace an occupied slot, or append at the fleet's first empty slot."""
        if not 0 <= index < self.max_fleet_size:
            return False

        self.model.replace_ship(index, name, cost)
        slot_rect = self.slot_rect(index)
        sprite_rect = sprite.get_rect(center=slot_rect.center)
        self.ships[index] = (sprite, name, cost, sprite_rect)
        return True

    def slot_rect(self, index):
        """Return the full clickable rectangle for a fleet grid position."""
        if not 0 <= index < self.max_fleet_size:
            raise IndexError("Fleet slot index out of range")
        row = index // self.icons_per_row
        col = index % self.icons_per_row
        return pygame.Rect(
            self.rect.x
            + self.left_offset
            + col * (self.icon_size[0] + self.spacing),
            self.rect.y + self.top_offset + row * self.icon_total_height,
            self.icon_size[0],
            self.icon_size[1],
        )

    def slot_index_at_pos(self, pos):
        """Return the grid position at ``pos``, including empty positions."""
        for index in range(self.max_fleet_size):
            if self.slot_rect(index).collidepoint(pos):
                return index
        return None

    def occupied_slots(self):
        """Return ``(slot_index, ship_view)`` pairs without compacting gaps."""
        return tuple(
            (index, ship)
            for index, ship in enumerate(self.ships)
            if ship is not None
        )

    def occupied_slot_at_pos(self, pos):
        """Return an occupied grid slot at ``pos``, using the full square."""
        index = self.slot_index_at_pos(pos)
        if index is None or self.ships[index] is None:
            return None
        return index, self.ships[index]

    def clear(self):
        self.model.clear()
        self.ships = [None] * self.max_fleet_size

    def remove_ship_at_pos(self, pos):
        for index in range(self.max_fleet_size):
            ship = self.ships[index]
            if ship is not None:
                _, _, _, rect = ship
                if rect.collidepoint(pos):
                    return self.remove_ship_at_index(index)
        return False

    def remove_ship_at_index(self, index):
        """Remove one ship without compacting fleet positions."""
        if not 0 <= index < self.max_fleet_size:
            return False
        if self.ships[index] is not None:
            self.ships[index] = None
            self.model.remove_ship(index)
            return True
        return False

    def _update_ship_positions(self):
        pass

    def get_total_cost(self):
        return self.model.total_cost

    def draw(self, screen, font, player_font=None):
        pygame.draw.rect(screen, FLEET_BOX_COLOR, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, FLEET_BORDER_WIDTH)

        title_text = font.render(
            f"{self.title} - cost: {self.get_total_cost()}", True, ui.WHITE
        )
        screen.blit(title_text, (self.rect.x + 10, self.rect.y + 10))

        for index in range(self.max_fleet_size):
            slot = self.slot_rect(index)
            pygame.draw.rect(screen, FLEET_SLOT_COLOR, slot)

        for ship in self.ships:
            if ship is not None:
                sprite, _, _, rect = ship
                screen.blit(sprite, rect)


class ShipSelectionFleet(Fleet):
    """Fleet presentation used while choosing the next combatants."""

    def draw(self, screen, font, player_font=None):
        pygame.draw.rect(screen, ui.BLACK, self.rect)

        title_text = font.render(
            f"{self.title} - cost: {self.get_total_cost()}", True, ui.WHITE
        )
        screen.blit(title_text, (self.rect.x + 10, self.rect.y + 10))

        for ship in self.ships:
            if ship is not None:
                sprite, _, _, rect = ship
                screen.blit(sprite, rect)

        pygame.draw.rect(screen, self.color, self.rect, FLEET_BORDER_WIDTH)


def ship_selection_hover_alpha(ticks):
    """Return a zero-to-opaque triangular pulse for hover outlines."""
    phase = (ticks % SHIP_SELECTION_HOVER_FADE_MS) / SHIP_SELECTION_HOVER_FADE_MS
    return round(255 * (1 - abs(2 * phase - 1)))


def draw_alpha_rect_outline(screen, rect, color, alpha, width):
    """Draw an alpha-blended rectangle outline on an opaque destination."""
    if alpha <= 0:
        return
    surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(surface, (*color, alpha), surface.get_rect(), width)
    screen.blit(surface, rect)
