import pygame
import json
import os
import sys
from src.UI import UI, UISlider, UIButton
import src.Const as Const

# Define constants
TITLE_FONT_SIZE = int(Const.SCREEN_HEIGHT*.08)
SETTINGS_FILE = Const.TRAINING_JSON_PATH

def load_settings():
    """Load settings from file or use defaults."""
    default_settings = Const.DEFAULT_TRAINING
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
    font = pygame.font.SysFont(None, int(0.03*Const.SCREEN_HEIGHT))
    settings = load_settings()
    background = UI.load_background(Const.MENU_BG_PATH, Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT)

    slider_left = int(Const.SCREEN_WIDTH*0.25)
    slider_width = int(Const.SCREEN_WIDTH*0.5)

    HH = [0] * 6
    for ii in range(6):
        HH[ii] = int(0.25*Const.SCREEN_HEIGHT+(ii * 0.09*Const.SCREEN_HEIGHT))

    sliders = [
        UISlider.Slider(slider_left, HH[0], slider_width, 0.0001, 0.01,
                        settings["learning_rate"], "Learning Rate", step=0.0001),
        UISlider.Slider(slider_left, HH[1], slider_width, 0.8, 1.0,
                        settings["discount_factor"], "Discount Factor", step=0.01),
        UISlider.Slider(slider_left, HH[2], slider_width, 0.0, 1.0,
                        settings["epsilon"], "Epsilon", step=0.01),
        UISlider.Slider(slider_left, HH[3], slider_width, 1, 20,
                        settings["number_of_hidden_layers"],"Number of Hidden Layers", is_int=True),
        UISlider.Slider(slider_left, HH[4], slider_width, 16, 512,
                        settings["layer_size"], "Layer Size", is_int=True, step=16),
        UISlider.Slider(slider_left, HH[5], slider_width, 32, 256,
                        settings["batch_size"], "Batch Size", is_int=True, step=32),
    ]
    back_to_menu = [False]

    def go_back():
        """Return to the main menu."""
        back_to_menu[0] = True

    def save_and_exit():
        """Save settings and return to the main menu."""
        new_settings = {
            slider.label.replace(" ", "_").lower(): slider.value
            for slider in sliders
        }
        save_settings(new_settings)
        back_to_menu[0] = True

    save_button = UIButton.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Save",
        save_and_exit,
        UI.OK_GREEN,
        UI.OK_GREEN_HI
    )

    cancel_button = UIButton.Button(
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
        clock.tick(Const.FPS)
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

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(UI.BG_COLOR)

        # Draw title using shared function
        UI.draw_title(screen, "Training Settings", TITLE_FONT_SIZE, int(0.1*Const.SCREEN_HEIGHT))

        # Draw UI elements
        for slider in sliders:
            slider.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
