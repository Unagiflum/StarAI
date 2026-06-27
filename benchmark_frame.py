import pygame
import os
import sys
import time

# Setup for headless so it doesn't pop up a window
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle.battle import BattleSimulation
from src.Objects.Ships.Earthling.Earthling import Earthling
from src.Objects.Ships.Yehat.Yehat import Yehat
from src.Battle.battle_draw import draw_battle
from src.Battle.battle import StarFieldRenderer

def benchmark():
    # Use a typical resolution
    screen = pygame.display.set_mode((1536, 960))
    ship1 = Earthling("Earthling", 1)
    ship2 = Yehat("Yehat", 2)
    
    sim = BattleSimulation(screen, ship1, ship2)
    renderer = StarFieldRenderer()
    
    # Step the simulation a bit to get everything moving
    for _ in range(10):
        sim.step()
        
    def render_frame():
        draw_battle(
            screen,
            sim.world,
            sim.border_rect,
            sim.border_color,
            renderer,
            is_paused=False,
            interp_t=0.5
        )
        # Flip the display to include buffer swap time
        pygame.display.flip()
        
    print("Benchmarking full frame render time...")
    
    # Warmup
    for _ in range(10):
        render_frame()
        
    # Benchmark
    frames = 500
    start_time = time.perf_counter()
    
    for _ in range(frames):
        render_frame()
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    ms_per_frame = (total_time / frames) * 1000
    
    print(f"Total time for {frames} frames: {total_time:.3f} seconds")
    print(f"Average render time per frame: {ms_per_frame:.3f} ms")

if __name__ == "__main__":
    benchmark()
