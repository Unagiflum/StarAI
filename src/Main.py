import pygame
import sys
from src.Menus import PickFleet, TrainSettings, GameSettings
from src.UI import UI, UIButton
import src.Const as GameConstants

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
    UI.sound_manager.load_sounds()
    UI.sound_manager.set_volume(0.30)

    screen = pygame.display.set_mode((UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT))
    pygame.display.set_caption("StarAI")
    clock = pygame.time.Clock()
    background = UI.load_background(GameConstants.MAIN_BG_PATH, UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)

    # Create menu buttons
    button_width = int(0.3*UI.SCREEN_WIDTH)
    button_height = int(0.0625*UI.SCREEN_HEIGHT)
    start_y = int(UI.SCREEN_HEIGHT *0.35)
    y_spacing = int(0.075 * UI.SCREEN_HEIGHT)

    menu_items = [
        ("Play Game", PickFleet),
        ("Game Settings", GameSettings),
        ("Training Settings", TrainSettings),
        ("Quit", None)
    ]

    buttons = []
    for i, (text, module) in enumerate(menu_items):
        button = UIButton.Button(
            x=int(UI.SCREEN_WIDTH // 2 - button_width // 2),
            y=start_y + i * y_spacing,
            width=button_width,
            height=button_height,
            text=text,
            callback=lambda m=module: handle_menu_selection(m, screen),
            bg_color=UI.MAIN_BUTTON_COLOR,
            hover_color=UI.MAIN_BUTTON_COLOR_HI
        )
        buttons.append(button)

    running = True
    while running:
        clock.tick(UI.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle button events
            for button in buttons:
                button.handle_event(event, UI.sound_manager)

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(UI.BG_COLOR)

        # Draw title
        #title_font = pygame.font.SysFont(None, int(UI.SCREEN_HEIGHT * 0.1))
        UI.draw_title(screen, "StarAI", int(UI.SCREEN_HEIGHT * 0.15), UI.SCREEN_HEIGHT // 6)

        # Draw buttons
        button_font = pygame.font.SysFont(None, int(UI.SCREEN_HEIGHT * 0.05))
        for button in buttons:
            button.draw(screen, button_font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
