import pygame
import sys
from src.UI import ui, ui_slider, ui_button
import src.const as Const
from src.configuration import TrainingSettingsRepository
from src.persistence import PersistenceValidationError

# Define constants
TITLE_FONT_SIZE = int(Const.SCREEN_HEIGHT*.08)
SETTINGS_FILE = Const.TRAINING_JSON_PATH


def _settings_repository():
    return TrainingSettingsRepository(SETTINGS_FILE, Const.DEFAULT_TRAINING)

def load_settings():
    """Load settings from file or use defaults."""
    return _settings_repository().load().to_dict()

def save_settings(settings):
    """Save settings to file."""
    try:
        repository = _settings_repository()
        typed_settings = repository.codec.decode(settings)
        repository.save(typed_settings)
        print("Settings saved successfully.")
    except (OSError, PersistenceValidationError) as e:
        print(f"Error saving settings: {e}")

def run(screen):
    """Run the Training Settings menu."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(0.03*Const.SCREEN_HEIGHT))
    settings = load_settings()
    background = ui.load_background(Const.MENU_BG_PATH, Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT)

    slider_left = int(Const.SCREEN_WIDTH*0.25)
    slider_width = int(Const.SCREEN_WIDTH*0.5)

    HH = [0] * 6
    for ii in range(6):
        HH[ii] = int(0.25*Const.SCREEN_HEIGHT+(ii * 0.09*Const.SCREEN_HEIGHT))

    sliders = [
        ui_slider.Slider(slider_left, HH[0], slider_width, 0.0001, 0.01,
                        settings["learning_rate"], "Learning Rate", step=0.0001),
        ui_slider.Slider(slider_left, HH[1], slider_width, 0.8, 1.0,
                        settings["discount_factor"], "Discount Factor", step=0.01),
        ui_slider.Slider(slider_left, HH[2], slider_width, 0.0, 1.0,
                        settings["epsilon"], "Epsilon", step=0.01),
        ui_slider.Slider(slider_left, HH[3], slider_width, 1, 20,
                        settings["number_of_hidden_layers"],"Number of Hidden Layers", is_int=True),
        ui_slider.Slider(slider_left, HH[4], slider_width, 16, 512,
                        settings["layer_size"], "Layer Size", is_int=True, step=16),
        ui_slider.Slider(slider_left, HH[5], slider_width, 32, 256,
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

    save_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Save",
        save_and_exit,
        ui.OK_GREEN,
        ui.OK_GREEN_HI
    )

    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        go_back,
        ui.CAN_RED,
        ui.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(Const.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            for slider in sliders:
                slider.handle_event(event, ui.sound_manager)
            save_button.handle_event(event, ui.sound_manager)
            cancel_button.handle_event(event, ui.sound_manager)

        if back_to_menu[0]:
            return

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        # Draw title using shared function
        ui.draw_title(screen, "Training Settings", TITLE_FONT_SIZE, int(0.1*Const.SCREEN_HEIGHT))

        # Draw UI elements
        for slider in sliders:
            slider.draw(screen, font)
        save_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
