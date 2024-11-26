import pygame
import json
import os
import sys
from UI import UI

# Define constants
TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT*.08)
SETTINGS_FILE = "Config/Trainingsettings.json"

def load_settings():
    """Load settings from file or use defaults."""
    default_settings = {
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "number_of_layers": 3,  # Updated key name
        "layer_size": 128,
        "batch_size": 64,
    }
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
            return {**default_settings, **settings}
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            return default_settings
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
    """Run the Training Settings menu."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(0.03*UI.SCREEN_HEIGHT))
    settings = load_settings()

    slider_left = int(UI.SCREEN_WIDTH*0.25)
    slider_width = int(UI.SCREEN_WIDTH*0.5)

    HH = [0] * 6
    for ii in range(6):
        HH[ii] = int(0.25*UI.SCREEN_HEIGHT+(ii * 0.1*UI.SCREEN_HEIGHT))

    sliders = [
        UI.Slider(slider_left, HH[0], slider_width, 0.0001, 0.01, settings["learning_rate"], "Learning Rate"),
        UI.Slider(slider_left, HH[1], slider_width, 0.8, 1.0, settings["discount_factor"], "Discount Factor"),
        UI.Slider(slider_left, HH[2], slider_width, 0.0, 1.0, settings["epsilon"], "Epsilon"),
        UI.Slider(slider_left, HH[3], slider_width, 1, 20, settings["number_of_layers"], "Number of Hidden Layers", is_int=True),  # Updated key
        UI.Slider(slider_left, HH[4], slider_width, 16, 512, settings["layer_size"], "Layer Size", is_int=True, step=16),
        UI.Slider(slider_left, HH[5], slider_width, 32, 256, settings["batch_size"], "Batch Size", is_int=True, step=32),
    ]

    back_to_menu = [False]

    def go_back():
        """Return to the main menu."""
        UI.sound_manager.play_sound('menu')
        back_to_menu[0] = True

    def save_and_exit():
        """Save settings and return to the main menu."""
        UI.sound_manager.play_sound('menu')
        new_settings = {
            slider.label.replace(" ", "_").lower(): slider.value
            for slider in sliders
        }
        save_settings(new_settings)
        back_to_menu[0] = True

    save_button = UI.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Save",
        save_and_exit,
        UI.OK_GREEN,
        UI.OK_GREEN_HI
    )

    cancel_button = UI.Button(
        UI.can_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Cancel",
        go_back,
        UI.CAN_RED,
        UI.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(UI.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for slider in sliders:
                slider.handle_event(event, UI.sound_manager)
            save_button.handle_event(event, UI.sound_manager)
            cancel_button.handle_event(event, UI.sound_manager)

        if back_to_menu[0]:
            return

        screen.fill(UI.BG_COLOR)

        # Draw title using shared function
        UI.draw_title(screen, "Training Settings", TITLE_FONT_SIZE, int(0.1*UI.SCREEN_HEIGHT))

        # Draw UI elements
        for slider in sliders:
            slider.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
