import pygame
import json
import os
import sys
from src.UI import UI, UIButton
import src.Const as GameConstants

TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT*.08)
SETTINGS_FILE = GameConstants.GAME_JSON_PATH

def load_settings():
    """Load settings from file or use defaults."""
    default_settings = UI.DEFAULT_KEYS
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Convert saved integer values back to readable names
                settings = {key: pygame.key.name(value) for key, value in settings.items()}
            return {**default_settings, **settings}  # Merge defaults and loaded settings
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            return {key: pygame.key.name(value) for key, value in default_settings.items()}
    else:
        return {key: pygame.key.name(value) for key, value in default_settings.items()}


def save_settings(settings):
    """Save settings to file."""
    try:
        # Convert human-readable key names to Pygame constants
        converted_settings = {
            key: pygame.key.key_code(value) if isinstance(value, str) else value
            for key, value in settings.items()
        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(converted_settings, f, indent=4)
        print("Settings saved successfully.")
    except Exception as e:
        print(f"Error saving settings: {e}")



def run(screen):
    """Run the Play Settings menu."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(0.03*UI.SCREEN_HEIGHT))
    settings = load_settings()
    background = UI.load_background(GameConstants.MENU_BG_PATH, UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)

    # Create key bindings
    key_bindings = []
    start_y = int(0.25*UI.SCREEN_HEIGHT)
    y_spacing = int(0.08*UI.SCREEN_HEIGHT)

    # Player 1 settings (left column)
    player1_labels = ["Player 1: Left", "Player 1: Right", "Player 1: Forward", "Player 1: Action 1",
                      "Player 1: Action 2"]
    for i, label in enumerate(player1_labels):
        key_bindings.append(
            UIButton.KeyBinding(
                int(0.25*UI.SCREEN_WIDTH),
                start_y + i * y_spacing,
                int(0.2*UI.SCREEN_WIDTH),
                int(0.05*UI.SCREEN_HEIGHT),
                label,
                settings[label]
            )
        )

    # Player 2 settings (right column)
    player2_labels = ["Player 2: Left", "Player 2: Right", "Player 2: Forward", "Player 2: Action 1",
                      "Player 2: Action 2"]
    for i, label in enumerate(player2_labels):
        key_bindings.append(
            UIButton.KeyBinding(
                int(0.65*UI.SCREEN_WIDTH),
                start_y + i * y_spacing,
                int(0.2*UI.SCREEN_WIDTH),
                int(0.05*UI.SCREEN_HEIGHT),
                label,
                settings[label]
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
    save_button = UIButton.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        text="Save",
        callback=save_and_exit,
        bg_color=UI.OK_GREEN,
        hover_color=UI.OK_GREEN_HI
    )

    cancel_button = UIButton.Button(
        UI.can_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        text="Cancel",
        callback=go_back,
        bg_color=UI.CAN_RED,
        hover_color=UI.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(UI.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle events for key bindings and buttons
            for binding in key_bindings:
                binding.handle_event(event, UI.sound_manager)
            save_button.handle_event(event, UI.sound_manager)
            cancel_button.handle_event(event, UI.sound_manager)

        # Check if we should return to the main menu
        if back_to_menu[0]:
            return

        # Draw screen
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(UI.BG_COLOR)

        # Draw title
        UI.draw_title(screen, "Game Settings", TITLE_FONT_SIZE, 0.1*UI.SCREEN_HEIGHT)

        # Draw key bindings
        for binding in key_bindings:
            binding.draw(screen, font)

        # Draw buttons
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()