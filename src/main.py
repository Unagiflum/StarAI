import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pygame
from src.Menus import pick_fleet, train_settings, game_settings
from src.UI import ui, ui_button
from src.audio import PygameAudioService
from src.resources import default_assets
import src.const as const

# Warning overlay timing (in frames at FPS).
_WARNING_VISIBLE_FRAMES = const.FPS * 5  # fully visible
_WARNING_FADE_FRAMES = const.FPS * 2  # fade-out
_WARNING_TOTAL_FRAMES = _WARNING_VISIBLE_FRAMES + _WARNING_FADE_FRAMES


def _build_warning_surface(asset_errors):
    """Build a transparent surface summarising preload failures."""
    font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.03))
    title_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.04))

    count = len(asset_errors)
    categories = sorted({error.category for error in asset_errors})
    lines = [
        f"Warning: {count} asset{'s' if count != 1 else ''} failed to load.",
        f"Categories: {', '.join(categories)}",
        "Some objects may appear as colored shapes.",
    ]

    padding = 16
    rendered = [title_font.render(lines[0], True, (255, 200, 100))]
    rendered.extend(font.render(line, True, (220, 220, 220)) for line in lines[1:])

    width = max(r.get_width() for r in rendered) + padding * 2
    height = sum(r.get_height() + 4 for r in rendered) + padding * 2
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(
        surface,
        (20, 0, 0, 200),
        surface.get_rect(),
        border_radius=8,
    )
    pygame.draw.rect(
        surface,
        (255, 80, 80, 180),
        surface.get_rect(),
        2,
        border_radius=8,
    )
    y = padding
    for r in rendered:
        surface.blit(r, (padding, y))
        y += r.get_height() + 4
    return surface


def handle_menu_selection(
    menu_callable, screen, menu_sound_manager=None, audio_service=None
):
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

    screen = pygame.display.set_mode((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
    pygame.display.set_caption("StarAI")
    clock = pygame.time.Clock()

    # Preload all assets before anything else renders.
    asset_errors = default_assets().preload_all()
    if asset_errors:
        for error in asset_errors:
            print(
                f"Asset load failed [{error.category}] {error.name}: "
                f"{error.message}  ({error.path})"
            )

    warning_surface = _build_warning_surface(asset_errors) if asset_errors else None
    warning_frame = 0

    audio_service = PygameAudioService()
    menu_sound_manager = ui.SoundManager(audio_service=audio_service)
    menu_sound_manager.load_sounds()
    menu_sound_manager.set_volume(0.30)

    background = ui.load_background(
        const.MAIN_BG_PATH,
        const.SCREEN_WIDTH,
        const.SCREEN_HEIGHT,
    )

    # Create menu buttons
    button_width = int(0.3 * const.SCREEN_WIDTH)
    button_height = int(0.0625 * const.SCREEN_HEIGHT)
    start_y = int(const.SCREEN_HEIGHT * 0.35)
    y_spacing = int(0.075 * const.SCREEN_HEIGHT)

    menu_items = [
        ("Play Game", pick_fleet.run),
        ("Game Settings", game_settings.run),
        ("Training Settings", train_settings.run),
        ("Quit", None),
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
            hover_color=ui.MAIN_BUTTON_COLOR_HI,
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
        ui.draw_title(
            screen, "StarAI", int(const.SCREEN_HEIGHT * 0.15), const.SCREEN_HEIGHT // 6
        )

        # Draw buttons
        button_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.05))
        for button in buttons:
            button.draw(screen, button_font)

        # Draw fading warning overlay if assets failed to load.
        if warning_surface is not None and warning_frame < _WARNING_TOTAL_FRAMES:
            if warning_frame < _WARNING_VISIBLE_FRAMES:
                alpha = 255
            else:
                fade_progress = (
                    warning_frame - _WARNING_VISIBLE_FRAMES
                ) / _WARNING_FADE_FRAMES
                alpha = max(0, int(255 * (1.0 - fade_progress)))
            overlay = warning_surface.copy()
            overlay.set_alpha(alpha)
            x = (const.SCREEN_WIDTH - overlay.get_width()) // 2
            y = const.SCREEN_HEIGHT - overlay.get_height() - 20
            screen.blit(overlay, (x, y))
            warning_frame += 1

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
