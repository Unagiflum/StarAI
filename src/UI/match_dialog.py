import sys

import pygame

import src.const as const
from src.UI import ui, ui_button
from src.frame_timing import PresentationClock


SCRIM_ALPHA = 180
PANEL_COLOR = (35, 35, 35)
PANEL_BORDER_COLOR = (190, 190, 190)


class EndMatchDialog:
    """Opaque end-match confirmation displayed over a darkened snapshot."""

    def __init__(self, screen):
        screen_rect = screen.get_rect()
        panel_width = min(int(screen_rect.width * 0.56), 800)
        panel_height = min(int(screen_rect.height * 0.30), 290)
        self.panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        self.panel_rect.center = screen_rect.center
        self.scrim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        self.scrim.fill((0, 0, 0, SCRIM_ALPHA))

        # This surface intentionally has no per-pixel alpha: the popup itself
        # must remain fully opaque above the translucent background scrim.
        self.panel = pygame.Surface(self.panel_rect.size)
        self.panel.fill(PANEL_COLOR)
        pygame.draw.rect(
            self.panel,
            PANEL_BORDER_COLOR,
            self.panel.get_rect(),
            3,
            border_radius=8,
        )

        button_width = int(panel_width * 0.40)
        button_height = max(48, int(panel_height * 0.20))
        button_gap = int(panel_width * 0.05)
        button_y = self.panel_rect.bottom - button_height - int(panel_height * 0.12)
        group_left = self.panel_rect.centerx - button_gap // 2 - button_width

        self.confirm_button = ui_button.Button(
            group_left,
            button_y,
            button_width,
            button_height,
            "Confirm (end match)",
            None,
            bg_color=ui.CAN_RED_HI,
            hover_color=(220, 45, 45, 255),
        )
        self.cancel_button = ui_button.Button(
            group_left + button_width + button_gap,
            button_y,
            button_width,
            button_height,
            "Cancel (continue match)",
            None,
            bg_color=(70, 110, 70, 255),
            hover_color=(90, 155, 90, 255),
        )
        self.title_font = pygame.font.SysFont(None, max(42, int(panel_height * 0.22)))
        self.button_font = pygame.font.SysFont(None, max(28, int(button_height * 0.55)))

    def handle_event(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return None
        if self.confirm_button.rect.collidepoint(event.pos):
            return True
        if self.cancel_button.rect.collidepoint(event.pos):
            return False
        return None

    def draw(self, screen, background):
        screen.blit(background, (0, 0))
        screen.blit(self.scrim, (0, 0))
        screen.blit(self.panel, self.panel_rect)

        title = self.title_font.render("End Match?", True, ui.WHITE)
        title_rect = title.get_rect(
            centerx=self.panel_rect.centerx,
            top=self.panel_rect.top + int(self.panel_rect.height * 0.12),
        )
        screen.blit(title, title_rect)

        self.confirm_button.draw(screen, self.button_font)
        self.cancel_button.draw(screen, self.button_font)


def confirm_end_match(screen, menu_sound_manager=None):
    """Block on an end-match prompt and return True only when confirmed."""
    background = screen.copy()
    dialog = EndMatchDialog(screen)
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            decision = dialog.handle_event(event)
            if decision is not None:
                if menu_sound_manager:
                    menu_sound_manager.play_sound("menu")
                return decision

        dialog.draw(screen, background)
        pygame.display.flip()
        clock.tick()
