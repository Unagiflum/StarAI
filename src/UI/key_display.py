"""Rendering helpers for the live keyboard-input display."""

import pygame

from . import ui


KEY_ABBREVIATIONS = {
    pygame.K_BACKSPACE: "Backspace",
    pygame.K_TAB: "Tab",
    pygame.K_RETURN: "Enter",
    pygame.K_ESCAPE: "Esc",
    pygame.K_SPACE: "Space",
    pygame.K_DELETE: "Del",
    pygame.K_INSERT: "Ins",
    pygame.K_PAGEUP: "PgUp",
    pygame.K_PAGEDOWN: "PgDn",
    pygame.K_PRINTSCREEN: "PrtSc",
    pygame.K_SCROLLOCK: "ScrLk",
    pygame.K_NUMLOCK: "NumLk",
    pygame.K_CAPSLOCK: "Caps Lock",
    pygame.K_LSHIFT: "L Shift",
    pygame.K_RSHIFT: "R Shift",
    pygame.K_LCTRL: "L Ctrl",
    pygame.K_RCTRL: "R Ctrl",
    pygame.K_LALT: "L Alt",
    pygame.K_RALT: "R Alt",
    pygame.K_LGUI: "L Meta",
    pygame.K_RGUI: "R Meta",
    pygame.K_KP0: "Num 0",
    pygame.K_KP1: "Num 1",
    pygame.K_KP2: "Num 2",
    pygame.K_KP3: "Num 3",
    pygame.K_KP4: "Num 4",
    pygame.K_KP5: "Num 5",
    pygame.K_KP6: "Num 6",
    pygame.K_KP7: "Num 7",
    pygame.K_KP8: "Num 8",
    pygame.K_KP9: "Num 9",
    pygame.K_KP_PERIOD: "Num .",
    pygame.K_KP_DIVIDE: "Num /",
    pygame.K_KP_MULTIPLY: "Num *",
    pygame.K_KP_MINUS: "Num -",
    pygame.K_KP_PLUS: "Num +",
    pygame.K_KP_ENTER: "Num Enter",
    pygame.K_KP_EQUALS: "Num =",
}


def standard_key_abbreviation(key):
    """Return a compact, conventional label for a Pygame key code."""
    if key in KEY_ABBREVIATIONS:
        return KEY_ABBREVIATIONS[key]
    name = pygame.key.name(key)
    if len(name) == 1 or (name.startswith("f") and name[1:].isdigit()):
        return name.upper()
    return name.title()


def draw_pressed_keys(surface, key_codes, panel_rect, font, label=None):
    """Draw held keys as white keycaps in a gray panel.

    Returns the keycap rectangles to make layout behavior observable to tests.
    """
    pygame.draw.rect(surface, ui.GREY, panel_rect, border_radius=6)
    padding = max(6, int(panel_rect.height * 0.1))
    gap = max(5, int(panel_rect.height * 0.06))
    content_top = panel_rect.top + padding
    if label:
        label_surface = font.render(label, True, ui.BLACK)
        label_rect = label_surface.get_rect(
            midtop=(panel_rect.centerx, content_top)
        )
        surface.blit(label_surface, label_rect)
        content_top = label_rect.bottom + gap

    labels = [standard_key_abbreviation(key) for key in key_codes]
    if not labels:
        pygame.draw.rect(
            surface,
            ui.BLACK,
            panel_rect,
            3,
            border_radius=6,
        )
        return ()

    key_height = font.get_height() + 12
    max_row_width = max(1, panel_rect.width - 2 * padding)
    items = []
    for label in labels:
        text = font.render(label, True, ui.BLACK)
        width = max(key_height, text.get_width() + 20)
        items.append((text, min(width, max_row_width)))

    rows = []
    row = []
    row_width = 0
    for item in items:
        required_width = item[1] if not row else gap + item[1]
        if row and row_width + required_width > max_row_width:
            rows.append((row, row_width))
            row = []
            row_width = 0
            required_width = item[1]
        row.append(item)
        row_width += required_width
    rows.append((row, row_width))

    rows_height = len(rows) * key_height + (len(rows) - 1) * gap
    content_bottom = panel_rect.bottom - padding
    y = content_top + max(0, (content_bottom - content_top - rows_height) // 2)
    key_rects = []
    previous_clip = surface.get_clip()
    surface.set_clip(panel_rect)
    for row, row_width in rows:
        x = panel_rect.centerx - row_width // 2
        for text, width in row:
            key_rect = pygame.Rect(x, y, width, key_height)
            pygame.draw.rect(surface, ui.WHITE, key_rect, border_radius=6)
            surface.blit(text, text.get_rect(center=key_rect.center))
            key_rects.append(key_rect)
            x += width + gap
        y += key_height + gap
    surface.set_clip(previous_clip)
    pygame.draw.rect(
        surface,
        ui.BLACK,
        panel_rect,
        3,
        border_radius=6,
    )
    return tuple(key_rects)
