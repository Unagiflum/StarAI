import pygame
import sys
from src.UI import UI


def run(screen, ship1, ship2):
    """
    Run the battle simulation between two ships.

    Args:
        screen (pygame.Surface): The main game display surface
        ship1 (SpaceShip): The first player's selected ship
        ship2 (SpaceShip): The second player's selected ship
    """
    clock = pygame.time.Clock()
    background = UI.load_background("UI/Menu.png", UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)
    font = pygame.font.SysFont(None, int(UI.SCREEN_HEIGHT * 0.05))

    # Placeholder battle loop
    running = True
    while running:
        clock.tick(UI.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(UI.BG_COLOR)

        # Draw placeholder text
        ship1_text = f"Ship 1: {ship1.name}"
        ship2_text = f"Ship 2: {ship2.name}"
        text1 = font.render(ship1_text, True, UI.WHITE)
        text2 = font.render(ship2_text, True, UI.WHITE)

        screen.blit(text1, (UI.SCREEN_WIDTH // 4, UI.SCREEN_HEIGHT // 2))
        screen.blit(text2, (3 * UI.SCREEN_WIDTH // 4, UI.SCREEN_HEIGHT // 2))

        pygame.display.flip()