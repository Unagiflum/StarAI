import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
from src.Menus import pick_fleet, train_settings, game_settings
from src.UI import ui, ui_button
from src.audio import PygameAudioService
import src.const as const


def handle_menu_selection(
        menu_callable, screen, menu_sound_manager=None, audio_service=None):
    """Handle the selected menu item."""
    if menu_callable is None:
        pygame.quit()
        sys.exit()
    menu_callable(
        screen=screen,
        menu_sound_manager=menu_sound_manager,
        audio_service=audio_service,
    )


def main():
    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()
    audio_service = PygameAudioService()
    menu_sound_manager = ui.SoundManager(audio_service=audio_service)
    menu_sound_manager.load_sounds()
    menu_sound_manager.set_volume(0.30)

    screen = pygame.display.set_mode((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
    pygame.display.set_caption("StarAI")
    clock = pygame.time.Clock()
    background = ui.load_background(const.MAIN_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT)

    # Create menu buttons
    button_width = int(0.3 * const.SCREEN_WIDTH)
    button_height = int(0.0625 * const.SCREEN_HEIGHT)
    start_y = int(const.SCREEN_HEIGHT * 0.35)
    y_spacing = int(0.075 * const.SCREEN_HEIGHT)

    menu_items = [
        ("Play Game", pick_fleet.run),
        ("Game Settings", game_settings.run),
        ("Training Settings", train_settings.run),
        ("Quit", None)
    ]

    buttons = []
    for i, (text, menu_callable) in enumerate(menu_items):
        button = ui_button.Button(
            x=int(const.SCREEN_WIDTH // 2 - button_width // 2),
            y=start_y + i * y_spacing,
            width=button_width,
            height=button_height,
            text=text,
            callback=lambda selected=menu_callable: handle_menu_selection(
                selected,
                screen,
                menu_sound_manager=menu_sound_manager,
                audio_service=audio_service,
            ),
            bg_color=ui.MAIN_BUTTON_COLOR,
            hover_color=ui.MAIN_BUTTON_COLOR_HI
        )
        buttons.append(button)

    running = True
    while running:
        clock.tick(const.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle button events
            for button in buttons:
                button.handle_event(event, menu_sound_manager)

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        # Draw title
        # title_font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.1))
        ui.draw_title(screen, "StarAI", int(const.SCREEN_HEIGHT * 0.15), const.SCREEN_HEIGHT // 6)

        # Draw buttons
        button_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.05))
        for button in buttons:
            button.draw(screen, button_font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
