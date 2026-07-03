import sys

import pygame

import src.const as const
from src.UI import ui, ui_button, ui_slider
from src.configuration import DisplaySettingsRepository
from src.frame_timing import PresentationClock
from src.persistence import PersistenceValidationError
from src.resources import default_assets


TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.08)
SETTINGS_FILE = const.DISPLAY_JSON_PATH
CROSSHAIR_OPTIONS = (
    ("Never", "never"),
    ("Mirror Match Only", "mirror_match_only"),
    ("Always", "always"),
)


def _settings_repository():
    return DisplaySettingsRepository(SETTINGS_FILE, const.DEFAULT_DISPLAY)


def load_settings():
    return _settings_repository().load()


def save_settings(values):
    repository = _settings_repository()
    settings = repository.codec.decode(values)
    repository.save(settings)
    return settings


def run(screen, menu_sound_manager=None, audio_service=None):
    """Run the display settings menu."""
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, int(0.03 * const.SCREEN_HEIGHT))
    settings = load_settings()
    background = ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )

    panel_left = int(const.SCREEN_WIDTH * 0.24)
    panel_width = int(const.SCREEN_WIDTH * 0.52)
    panel_gap = int(const.SCREEN_HEIGHT * 0.012)
    panel_height = int(const.SCREEN_HEIGHT * 0.12)
    panel_rects = tuple(
        pygame.Rect(
            panel_left,
            int(const.SCREEN_HEIGHT * 0.20) + index * (panel_height + panel_gap),
            panel_width,
            panel_height,
        )
        for index in range(3)
    )
    panel_surfaces = []
    for panel_rect in panel_rects:
        panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            panel_surface,
            ui.SETTINGS_PANEL,
            panel_surface.get_rect(),
            border_radius=5,
        )
        panel_surfaces.append(panel_surface)
    control_left = int(const.SCREEN_WIDTH * 0.25)
    control_width = int(const.SCREEN_WIDTH * 0.5)
    frame_rate = ui_slider.Slider(
        control_left,
        panel_rects[0].y + int(const.SCREEN_HEIGHT * 0.02),
        control_width,
        24,
        120,
        settings.video_frame_rate,
        "Video Frame Rate",
        is_int=True,
        step=24,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI,
    )

    selected_crosshairs = [settings.ship_crosshairs]
    radio_buttons = []

    def select_crosshairs(value):
        selected_crosshairs[0] = value
        for button, (_, option_value) in zip(radio_buttons, CROSSHAIR_OPTIONS):
            button.selected = option_value == value

    radio_top = panel_rects[1].y + int(const.SCREEN_HEIGHT * 0.045)
    radio_width = control_width // len(CROSSHAIR_OPTIONS) - 8
    for index, (label, value) in enumerate(CROSSHAIR_OPTIONS):
        radio_buttons.append(
            ui_button.RadioButton(
                control_left + index * (radio_width + 12),
                radio_top,
                radio_width,
                int(const.SCREEN_HEIGHT * 0.06),
                label,
                lambda selected=value: select_crosshairs(selected),
                selected=value == selected_crosshairs[0],
            )
        )

    gravity_marker = ui_button.Checkbox(
        control_left,
        panel_rects[2].y + int(const.SCREEN_HEIGHT * 0.03),
        control_width,
        int(const.SCREEN_HEIGHT * 0.06),
        "Show Planet Gravity Marker",
        initial_state=settings.show_planet_gravity_marker,
    )
    back_to_menu = [False]

    def go_back():
        back_to_menu[0] = True

    def save_and_exit():
        values = {
            "video_frame_rate": int(frame_rate.value),
            "ship_crosshairs": selected_crosshairs[0],
            "show_planet_gravity_marker": gravity_marker.value,
        }
        try:
            new_settings = save_settings(values)
        except (OSError, PersistenceValidationError) as error:
            print(f"Error saving display settings: {error}")
            return

        frame_rate_changed = (
            new_settings.video_frame_rate != const.VIDEO_FPS
        )
        const.apply_display_settings(new_settings)
        if frame_rate_changed:
            default_assets().invalidate_interpolated_graphics()
        back_to_menu[0] = True

    save_button = ui_button.Button(
        ui.ok_button_left, ui.ok_button_top, ui.ok_button_width, ui.ok_button_height,
        "Save", save_and_exit, ui.OK_GREEN, ui.OK_GREEN_HI,
    )
    cancel_button = ui_button.Button(
        ui.can_button_left, ui.ok_button_top, ui.ok_button_width, ui.ok_button_height,
        "Cancel", go_back, ui.CAN_RED, ui.CAN_RED_HI,
    )

    while True:
        clock.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            frame_rate.handle_event(event, menu_sound_manager)
            for button in radio_buttons:
                button.handle_event(event, menu_sound_manager)
            gravity_marker.handle_event(event, menu_sound_manager)
            save_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

        if back_to_menu[0]:
            return

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)
        ui.draw_title(
            screen, "Display Settings", TITLE_FONT_SIZE,
            int(0.1 * const.SCREEN_HEIGHT),
        )
        for panel_rect, panel_surface in zip(panel_rects, panel_surfaces):
            screen.blit(panel_surface, panel_rect)
        frame_rate.draw(screen, font)

        label = font.render("Ship Color Crosshairs", True, ui.WHITE)
        screen.blit(label, (control_left, radio_top - label.get_height() - 8))
        for button in radio_buttons:
            button.draw(screen, font)
        gravity_marker.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)
        pygame.display.flip()
