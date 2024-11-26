import pygame
import json
import os
from UI import UI

TITLE_FONT_SIZE = 64
SETTINGS_FILE = "Config/Gamesettings.json"

def load_settings():
    """Load settings from file or use defaults."""
    default_settings = {
        "Player 1: Left": "a",
        "Player 1: Right": "d",
        "Player 1: Forward": "w",
        "Player 1: Action 1": "tab",
        "Player 1: Action 2": "`",
        "Player 2: Left": "left",
        "Player 2: Right": "right",
        "Player 2: Forward": "up",
        "Player 2: Action 1": "right ctrl",
        "Player 2: Action 2": "right shift"
    }
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
            return {**default_settings, **settings}  # Merge defaults and loaded settings
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            return default_settings
    else:
        return default_settings


def save_settings(settings):
    """Save settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print("Settings saved successfully.")
    except Exception as e:
        print(f"Error saving settings: {e}")


def run(screen):
    """Run the Play Settings menu."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    settings = load_settings()

    # Create key bindings
    key_bindings = []
    start_y = 250
    y_spacing = 60

    # Player 1 settings (left column)
    player1_labels = ["Player 1: Left", "Player 1: Right", "Player 1: Forward", "Player 1: Action 1",
                      "Player 1: Action 2"]
    for i, label in enumerate(player1_labels):
        key_bindings.append(UI.KeyBinding(250, start_y + i * y_spacing, 200, 40, label, settings[label]))

    # Player 2 settings (right column)
    player2_labels = ["Player 2: Left", "Player 2: Right", "Player 2: Forward", "Player 2: Action 1",
                      "Player 2: Action 2"]
    for i, label in enumerate(player2_labels):
        key_bindings.append(UI.KeyBinding(750, start_y + i * y_spacing, 200, 40, label, settings[label]))

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
    save_button = UI.Button(
        x=UI.SCREEN_WIDTH // 2 - 220,
        y=700,
        width=200,
        height=50,
        text="Save",
        callback=save_and_exit,
        bg_color=UI.OK_GREEN,
        hover_color=UI.OK_GREEN_HI
    )

    cancel_button = UI.Button(
        x=UI.SCREEN_WIDTH // 2 + 20,
        y=700,
        width=200,
        height=50,
        text="Cancel",
        callback=go_back,
        bg_color=UI.CAN_RED,
        hover_color=UI.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(60)
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
        screen.fill(UI.DARK_BLUE)

        # Draw title
        UI.draw_title(screen, "Game Settings", TITLE_FONT_SIZE, 80)

        # Draw key bindings
        for binding in key_bindings:
            binding.draw(screen, font)

        # Draw buttons
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()