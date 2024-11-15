import pygame
import sys

# Initialize PyGame
pygame.init()

# Set screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Spaceship Fighting Game")

# Game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Black background
    pygame.display.flip()
    clock.tick(30)  # 30 FPS

pygame.quit()
sys.exit()
