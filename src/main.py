import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(SRC_ROOT)

import pygame
from src.Menus import pick_fleet, train_settings, game_settings
from src.UI import ui, ui_button
import src.const as const


def handle_menu_selection(module, screen):
    """Handle the selected menu item."""
    if module is None:
        pygame.quit()
        sys.exit()
    try:
        if hasattr(module, 'run'):
            module.run(screen)
        else:
            print(f"Module '{module.__name__}' does not have a 'run' function. Continuing.")
    except Exception as e:
        print(f"An error occurred while running '{module.__name__}': {e}")


def main():
    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()
    ui.sound_manager.load_sounds()
    ui.sound_manager.set_volume(0.30)

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
        ("Play Game", pick_fleet),
        ("Game Settings", game_settings),
        ("Training Settings", train_settings),
        ("Quit", None)
    ]

    buttons = []
    for i, (text, module) in enumerate(menu_items):
        button = ui_button.Button(
            x=int(const.SCREEN_WIDTH // 2 - button_width // 2),
            y=start_y + i * y_spacing,
            width=button_width,
            height=button_height,
            text=text,
            callback=lambda m=module: handle_menu_selection(m, screen),
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
                button.handle_event(event, ui.sound_manager)

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
