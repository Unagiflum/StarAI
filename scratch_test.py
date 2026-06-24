import pygame
import math
import os

pygame.init()
screen = pygame.display.set_mode((400, 400))
screen.fill((0, 0, 0))

radius = 100
surf_size = radius * 2 + 12
circle_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
rect = pygame.Rect(6, 6, radius * 2, radius * 2)
color = (255, 0, 0, 128)

# Center of the surface
cx, cy = surf_size // 2, surf_size // 2

angles = [45, 135, 225, 315]
dot_radius = 3 # 6px thickness

for angle_deg in angles:
    angle_rad = math.radians(angle_deg)
    # Pygame arc goes counter-clockwise, 0 is right.
    # We want top-left, top-right, bottom-left, bottom-right. 
    # x = cx + radius * math.cos(angle)
    # y = cy - radius * math.sin(angle)  # minus because y is down
    x = cx + radius * math.cos(angle_rad)
    y = cy - radius * math.sin(angle_rad)
    pygame.draw.circle(circle_surf, color, (int(x), int(y)), dot_radius)

screen.blit(circle_surf, (100, 100))

# Also draw using arc for comparison
circle_surf2 = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
arc_width_rad = math.radians(5) # 5 degrees wide
for angle_deg in angles:
    center_rad = math.radians(angle_deg)
    start_angle = center_rad - arc_width_rad / 2
    end_angle = center_rad + arc_width_rad / 2
    pygame.draw.arc(circle_surf2, (0, 255, 0, 128), rect, start_angle, end_angle, 6)

screen.blit(circle_surf2, (100, 100))

pygame.display.flip()
pygame.image.save(screen, "test_arc.png")
pygame.quit()
