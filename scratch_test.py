import pygame
import os
import sys

# Setup for headless
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle.battle import BattleSimulation
from src.Objects.Ships.Earthling.Earthling import Earthling
from src.Objects.Ships.Yehat.Yehat import Yehat
from src.Battle.battle_draw import draw_battle
from src.Battle.battle import StarFieldRenderer

def test_run():
    screen = pygame.display.set_mode((800, 600))
    ship1 = Earthling("Earthling", 1)
    ship2 = Yehat("Yehat", 2)
    
    sim = BattleSimulation(screen, ship1, ship2)
    renderer = StarFieldRenderer()
    
    print("Starting simulation...")
    for video_frame in range(30):
        if video_frame % 2 == 0:
            sim.step()
        draw_battle(
            screen,
            sim.world,
            sim.border_rect,
            sim.border_color,
            renderer,
            is_paused=False,
            interp_t=0.5
        )
        
    print("Simulation completed successfully.")

if __name__ == "__main__":
    test_run()
