import sys

import pygame

import src.const as const
from src.UI import ui, ui_button, ui_slider
from src.configuration import GameSettings, GameSettingsRepository
from src.resources import default_assets
from src.frame_timing import PresentationClock
from src.persistence import PersistenceValidationError


TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.08)
SETTINGS_FILE = const.GAME_JSON_PATH


def _settings_repository():
    return GameSettingsRepository(
        SETTINGS_FILE, const.DEFAULT_KEYS, const.DEFAULT_GAMEPLAY
    )


def load_settings():
    """Load key bindings from file or use defaults."""
    return _settings_repository().load().key_names()


def save_settings(settings):
    """Save key bindings to file."""
    try:
        repository = _settings_repository()
        typed_settings = repository.codec.from_key_names(
            settings, current=repository.load()
        )
        repository.save(typed_settings)
        print("Settings saved successfully.")
    except (OSError, PersistenceValidationError) as error:
        print(f"Error saving settings: {error}")


def _background():
    return ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )


def _draw_background(screen, background):
    if background:
        screen.blit(background, (0, 0))
    else:
        screen.fill(ui.BG_COLOR)


def _exit_on_quit(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()


def run_input_keys(screen, menu_sound_manager=None, audio_service=None):
    """Edit input keys, returning Save and Cancel to Game Settings."""
    _ = audio_service
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, int(0.03 * const.SCREEN_HEIGHT))
    settings = load_settings()
    background = _background()

    key_bindings = []
    start_y = int(0.25 * const.SCREEN_HEIGHT)
    y_spacing = int(0.08 * const.SCREEN_HEIGHT)
    player_labels = (
        (
            int(0.25 * const.SCREEN_WIDTH),
            const.P1_COLOR,
            (
                "Player 1: Left",
                "Player 1: Right",
                "Player 1: Forward",
                "Player 1: Action 1",
                "Player 1: Action 2",
            ),
        ),
        (
            int(0.65 * const.SCREEN_WIDTH),
            const.P2_COLOR,
            (
                "Player 2: Left",
                "Player 2: Right",
                "Player 2: Forward",
                "Player 2: Action 1",
                "Player 2: Action 2",
            ),
        ),
    )
    for x, player_color, labels in player_labels:
        for index, label in enumerate(labels):
            key_bindings.append(
                ui_button.KeyBinding(
                    x,
                    start_y + index * y_spacing,
                    int(0.2 * const.SCREEN_WIDTH),
                    int(0.05 * const.SCREEN_HEIGHT),
                    label,
                    settings[label],
                    bg_color=(*player_color, 75),
                    hover_color=(*player_color, 255),
                    border_color=player_color,
                )
            )

    finished = [False]

    def go_back():
        finished[0] = True

    def save_and_exit():
        save_settings({binding.label: binding.key for binding in key_bindings})
        finished[0] = True

    save_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Save",
        save_and_exit,
        ui.OK_GREEN,
        ui.OK_GREEN_HI,
    )
    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        go_back,
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )

    while not finished[0]:
        clock.tick()
        for event in pygame.event.get():
            _exit_on_quit(event)
            for binding in key_bindings:
                binding.handle_event(event, menu_sound_manager)
            save_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

        _draw_background(screen, background)
        ui.draw_title(
            screen, "Input Keys", TITLE_FONT_SIZE, int(0.1 * const.SCREEN_HEIGHT)
        )
        for binding in key_bindings:
            binding.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)
        pygame.display.flip()


def run_game_play(screen, menu_sound_manager=None, audio_service=None):
    """Edit settings that control battle gameplay."""
    _ = audio_service
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, int(0.03 * const.SCREEN_HEIGHT))
    settings = _settings_repository().load()
    background = _background()
    finished = [False]

    panel_left = int(const.SCREEN_WIDTH * 0.24)
    panel_width = int(const.SCREEN_WIDTH * 0.52)
    panel_gap = int(const.SCREEN_HEIGHT * 0.012)
    panel_height = int(const.SCREEN_HEIGHT * 0.12)
    panel_rects = tuple(
        pygame.Rect(
            panel_left,
            int(const.SCREEN_HEIGHT * 0.25)
            + index * (panel_height + panel_gap),
            panel_width,
            panel_height,
        )
        for index in range(2)
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
    asteroid_count = ui_slider.Slider(
        control_left,
        panel_rects[0].y + int(const.SCREEN_HEIGHT * 0.02),
        control_width,
        1,
        20,
        settings.asteroid_count,
        "Asteroid Count",
        is_int=True,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI,
    )
    ship_directions = ui_slider.Slider(
        control_left,
        panel_rects[1].y + int(const.SCREEN_HEIGHT * 0.02),
        control_width,
        16,
        64,
        settings.ship_directions,
        "Ship Directions",
        is_int=True,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI,
        values=(16, 32, 64),
    )

    def save_and_exit():
        new_settings = GameSettings(
            settings.bindings,
            asteroid_count=int(asteroid_count.value),
            ship_directions=int(ship_directions.value),
        )
        try:
            _settings_repository().save(new_settings)
        except (OSError, PersistenceValidationError) as error:
            print(f"Error saving game-play settings: {error}")
            return
        directions_changed = const.apply_game_settings(new_settings)
        if directions_changed:
            default_assets().invalidate_interpolated_graphics()
        finished[0] = True

    save_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Save",
        save_and_exit,
        ui.OK_GREEN,
        ui.OK_GREEN_HI,
    )
    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        lambda: finished.__setitem__(0, True),
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )

    while not finished[0]:
        clock.tick()
        for event in pygame.event.get():
            _exit_on_quit(event)
            asteroid_count.handle_event(event, menu_sound_manager)
            ship_directions.handle_event(event, menu_sound_manager)
            save_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

        _draw_background(screen, background)
        ui.draw_title(
            screen, "Game Play", TITLE_FONT_SIZE, int(0.1 * const.SCREEN_HEIGHT)
        )
        for panel_rect, panel_surface in zip(panel_rects, panel_surfaces):
            screen.blit(panel_surface, panel_rect)
        asteroid_count.draw(screen, font)
        ship_directions.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)
        pygame.display.flip()


def run(screen, menu_sound_manager=None, audio_service=None):
    """Run the Game Settings parent menu."""
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, int(0.04 * const.SCREEN_HEIGHT))
    background = _background()
    finished = [False]

    button_width = int(0.3 * const.SCREEN_WIDTH)
    button_height = int(0.0625 * const.SCREEN_HEIGHT)
    button_left = const.SCREEN_WIDTH // 2 - button_width // 2

    menu_buttons = (
        ui_button.Button(
            button_left,
            int(0.35 * const.SCREEN_HEIGHT),
            button_width,
            button_height,
            "Input Keys",
            lambda: run_input_keys(screen, menu_sound_manager, audio_service),
            ui.MENU_BUTTON_COLOR,
            ui.MENU_BUTTON_COLOR_HI,
        ),
        ui_button.Button(
            button_left,
            int(0.45 * const.SCREEN_HEIGHT),
            button_width,
            button_height,
            "Game Play",
            lambda: run_game_play(screen, menu_sound_manager, audio_service),
            ui.MENU_BUTTON_COLOR,
            ui.MENU_BUTTON_COLOR_HI,
        ),
    )
    back_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Back",
        lambda: finished.__setitem__(0, True),
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )

    while not finished[0]:
        clock.tick()
        for event in pygame.event.get():
            _exit_on_quit(event)
            for button in menu_buttons:
                button.handle_event(event, menu_sound_manager)
            back_button.handle_event(event, menu_sound_manager)

        _draw_background(screen, background)
        ui.draw_title(
            screen, "Game Settings", TITLE_FONT_SIZE, int(0.1 * const.SCREEN_HEIGHT)
        )
        for button in menu_buttons:
            button.draw(screen, font)
        back_button.draw(screen, font)
        pygame.display.flip()
