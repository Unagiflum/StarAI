import pygame
import sys
from src.UI import ui, ui_button
import src.const as const
from src.configuration import GameSettingsRepository
from src.persistence import PersistenceValidationError

TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.08)
SETTINGS_FILE = const.GAME_JSON_PATH


def _settings_repository():
    return GameSettingsRepository(SETTINGS_FILE, const.DEFAULT_KEYS)


def load_settings():
    """Load settings from file or use defaults."""
    return _settings_repository().load().key_names()


def save_settings(settings):
    """Save settings to file."""
    try:
        repository = _settings_repository()
        typed_settings = repository.codec.from_key_names(settings)
        repository.save(typed_settings)
        print("Settings saved successfully.")
    except (OSError, PersistenceValidationError) as e:
        print(f"Error saving settings: {e}")


def run(screen, menu_sound_manager=None, audio_service=None):
    """Run the Play Settings menu."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(0.03 * const.SCREEN_HEIGHT))
    settings = load_settings()
    background = ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )

    # Create key bindings
    key_bindings = []
    start_y = int(0.25 * const.SCREEN_HEIGHT)
    y_spacing = int(0.08 * const.SCREEN_HEIGHT)

    # Player 1 settings (left column)
    player1_labels = [
        "Player 1: Left",
        "Player 1: Right",
        "Player 1: Forward",
        "Player 1: Action 1",
        "Player 1: Action 2",
    ]
    for i, label in enumerate(player1_labels):
        key_bindings.append(
            ui_button.KeyBinding(
                int(0.25 * const.SCREEN_WIDTH),
                start_y + i * y_spacing,
                int(0.2 * const.SCREEN_WIDTH),
                int(0.05 * const.SCREEN_HEIGHT),
                label,
                settings[label],
            )
        )

    # Player 2 settings (right column)
    player2_labels = [
        "Player 2: Left",
        "Player 2: Right",
        "Player 2: Forward",
        "Player 2: Action 1",
        "Player 2: Action 2",
    ]
    for i, label in enumerate(player2_labels):
        key_bindings.append(
            ui_button.KeyBinding(
                int(0.65 * const.SCREEN_WIDTH),
                start_y + i * y_spacing,
                int(0.2 * const.SCREEN_WIDTH),
                int(0.05 * const.SCREEN_HEIGHT),
                label,
                settings[label],
            )
        )

    # Flag to return to the main menu
    back_to_menu = [False]

    def go_back():
        """Return to the main menu without saving."""
        back_to_menu[0] = True

    def save_and_exit():
        """Save settings and return to the main menu."""
        new_settings = {binding.label: binding.key for binding in key_bindings}
        save_settings(new_settings)
        back_to_menu[0] = True

    # Create buttons
    save_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        text="Save",
        callback=save_and_exit,
        bg_color=ui.OK_GREEN,
        hover_color=ui.OK_GREEN_HI,
    )

    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        text="Cancel",
        callback=go_back,
        bg_color=ui.CAN_RED,
        hover_color=ui.CAN_RED_HI,
    )

    running = True
    while running:
        clock.tick(const.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle events for key bindings and buttons
            for binding in key_bindings:
                binding.handle_event(event, menu_sound_manager)
            save_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

        # Check if we should return to the main menu
        if back_to_menu[0]:
            return

        # Draw screen
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        # Draw title
        ui.draw_title(
            screen, "Game Settings", TITLE_FONT_SIZE, 0.1 * const.SCREEN_HEIGHT
        )

        # Draw key bindings
        for binding in key_bindings:
            binding.draw(screen, font)

        # Draw buttons
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
