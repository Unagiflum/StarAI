import pygame
import sys
import PickFleet
import PlaySettings
import TrainSettings
from UI import UI

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 30

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

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("StarAI")
    clock = pygame.time.Clock()
    background = UI.load_background("UI/Main.png", SCREEN_WIDTH, SCREEN_HEIGHT)

    # Create menu buttons
    button_width = 300
    button_height = 50
    start_y = SCREEN_HEIGHT // 2 - 100
    y_spacing = 60

    menu_items = [
        ("Play Game", PickFleet),
        ("Game Settings", PlaySettings),
        ("Training Settings", TrainSettings),
        ("Quit", None)
    ]

    buttons = []
    for i, (text, module) in enumerate(menu_items):
        button = UI.Button(
            x=SCREEN_WIDTH // 2 - button_width // 2,
            y=start_y + i * y_spacing,
            width=button_width,
            height=button_height,
            text=text,
            callback=lambda m=module: handle_menu_selection(m, screen),
            bg_color=UI.MENU_BUTTON_COLOR,
            hover_color=UI.MENU_BUTTON_COLOR_HI
        )
        buttons.append(button)

    running = True
    while running:
        clock.tick(FPS)

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
            screen.fill(UI.BLACK)

        # Draw title
        title_font = pygame.font.SysFont(None, int(SCREEN_HEIGHT * 0.1))
        UI.draw_title(screen, "StarAI", int(SCREEN_HEIGHT * 0.15), SCREEN_HEIGHT // 6)

        # Draw buttons
        button_font = pygame.font.SysFont(None, int(SCREEN_HEIGHT * 0.05))
        for button in buttons:
            button.draw(screen, button_font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
