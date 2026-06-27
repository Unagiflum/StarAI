import pygame
import timeit

pygame.init()
screen = pygame.display.set_mode((1536, 960))
star = pygame.Surface((16, 16), pygame.SRCALPHA)
star.fill((255, 255, 255, 200))

print("Benchmarking star blitting (ms per frame)...")
for n in [100, 500, 1000, 2000, 5000, 10000]:
    # Run 100 iterations of blitting `n` stars
    t = timeit.timeit(f"for _ in range({n}): screen.blit(star, (0,0))", globals=globals(), number=100)
    # Convert to ms per frame
    ms_per_frame = (t / 100) * 1000
    print(f"{n} stars: {ms_per_frame:.3f} ms")
