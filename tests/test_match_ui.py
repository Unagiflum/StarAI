import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle import battle
from src.Battle.battle_draw import (
    BAR_WIDTH,
    BattleDrawLayout,
    BattleDrawController,
    BattleDrawOptions,
    HUD_AI_LABEL_FONT_SIZE,
    MARINE_REGION_HEIGHT,
    RenderSnapshot,
    VIEWPORT_COLUMN_WIDTH,
    VIEWPORT_MARGIN,
    VIEWPORT_SIZE,
    create_play_battle_layout,
    draw_battle,
    draw_battle_arena,
)
from src.Menus.pick_ship import show_battle_countdown
from src.UI.match_dialog import EndMatchDialog


class ResumeCountdownTests(unittest.TestCase):
    def test_translucent_overlay_preserves_the_battle_frame(self):
        screen = pygame.Surface((320, 180))
        background = pygame.Surface(screen.get_size())
        background.fill((100, 120, 140))
        fake_clock = SimpleNamespace(
            tick=mock.Mock(return_value=1.0), reset=mock.Mock()
        )

        with (
            mock.patch(
                "src.Menus.pick_ship.PresentationClock", return_value=fake_clock
            ),
            mock.patch("src.Menus.pick_ship.pygame.mouse.set_pos") as set_pos,
            mock.patch("src.Menus.pick_ship.pygame.event.get", return_value=[]),
            mock.patch("src.Menus.pick_ship.pygame.display.flip"),
        ):
            show_battle_countdown(
                screen,
                steps=1,
                step_time=0.5,
                background=background,
                overlay_alpha=128,
            )

        pixel = screen.get_at((0, 0))[:3]
        self.assertTrue(48 <= pixel[0] <= 50)
        self.assertTrue(58 <= pixel[1] <= 60)
        self.assertTrue(68 <= pixel[2] <= 70)
        set_pos.assert_called_once_with((0, screen.get_height() - 1))

    def test_zero_step_countdown_still_moves_mouse_to_bottom_left(self):
        screen = pygame.Surface((320, 180))

        with (
            mock.patch("src.Menus.pick_ship.PresentationClock"),
            mock.patch("src.Menus.pick_ship.pygame.mouse.set_pos") as set_pos,
        ):
            show_battle_countdown(screen, steps=0, step_time=0.5)

        set_pos.assert_called_once_with((0, screen.get_height() - 1))


class EndMatchDialogTests(unittest.TestCase):
    def setUp(self):
        self.screen = pygame.Surface((800, 600))
        self.dialog = EndMatchDialog(self.screen)

    def test_confirm_and_cancel_buttons_return_their_decisions(self):
        confirm = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=self.dialog.confirm_button.rect.center,
        )
        cancel = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=self.dialog.cancel_button.rect.center,
        )

        self.assertIs(self.dialog.handle_event(confirm), True)
        self.assertIs(self.dialog.handle_event(cancel), False)
        self.assertEqual(self.dialog.confirm_button.text, "Confirm (end match)")
        self.assertEqual(self.dialog.cancel_button.text, "Cancel (continue match)")

    def test_panel_is_opaque_and_background_is_darkened(self):
        first_background = pygame.Surface(self.screen.get_size())
        first_background.fill((255, 0, 0))
        second_background = pygame.Surface(self.screen.get_size())
        second_background.fill((0, 0, 255))
        sample = (
            self.dialog.panel_rect.left + 12,
            self.dialog.panel_rect.top + 12,
        )

        self.dialog.draw(self.screen, first_background)
        first_panel_pixel = self.screen.get_at(sample)
        outside_pixel = self.screen.get_at((0, 0))[:3]
        self.dialog.draw(self.screen, second_background)
        second_panel_pixel = self.screen.get_at(sample)

        self.assertEqual(first_panel_pixel, second_panel_pixel)
        self.assertLess(outside_pixel[0], 255)


class BattleHudLayoutTests(unittest.TestCase):
    def _empty_snapshot(self, ships=(), live_ships=()):
        return RenderSnapshot(
            stars=(),
            planets=(),
            thrust_markers=(),
            asteroids=(),
            abilities=(),
            ships=tuple(ships),
            effects=(),
            live_ships=tuple(live_ships),
        )

    def _ship(self, player):
        return SimpleNamespace(
            player=player,
            name=f"Ship {player}",
            position=[500.0 + player * 100, 500.0],
            previous_position=[500.0 + player * 100, 500.0],
            current_hp=8,
            max_hp=10,
            current_energy=5,
            max_energy=10,
            boarded_marines=(),
            limpets_attached=0,
        )

    def test_shared_controller_does_not_flip_display(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        arena_rect = pygame.Rect(
            const.SCREEN_LEFT,
            0,
            const.SCREEN_HEIGHT,
            const.SCREEN_HEIGHT,
        )

        with (
            mock.patch("src.Battle.battle_draw._render_world_to_surface"),
            mock.patch("src.Battle.battle_draw.pygame.display.flip") as flip,
        ):
            BattleDrawController().draw(
                screen,
                [],
                create_play_battle_layout(arena_rect),
                (255, 255, 255),
                mock.Mock(),
                options=BattleDrawOptions(draw_huds=False),
            )

        flip.assert_not_called()

    def test_shared_controller_draws_supplied_play_arena_and_huds(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        screen.fill((20, 20, 20))
        arena_rect = pygame.Rect(
            const.SCREEN_LEFT,
            0,
            const.SCREEN_HEIGHT,
            const.SCREEN_HEIGHT,
        )

        with mock.patch("src.Battle.battle_draw._render_world_to_surface"):
            BattleDrawController().draw(
                screen,
                [],
                create_play_battle_layout(arena_rect),
                (255, 255, 255),
                mock.Mock(),
                options=BattleDrawOptions(draw_instructions=False),
            )

        self.assertEqual(screen.get_at(arena_rect.center)[:3], (0, 0, 0))
        self.assertNotEqual(screen.get_at((285, 100))[:3], (20, 20, 20))
        self.assertNotEqual(screen.get_at((1250, 100))[:3], (20, 20, 20))

    def test_shared_controller_draws_arena_into_non_play_rect(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        unchanged = (20, 21, 22)
        marker = (1, 2, 3)
        screen.fill(unchanged)
        arena_rect = pygame.Rect(
            const.SCREEN_WIDTH - const.SCREEN_HEIGHT,
            0,
            const.SCREEN_HEIGHT,
            const.SCREEN_HEIGHT,
        )

        def fake_render(surface, *args, **kwargs):
            pygame.draw.rect(
                surface,
                marker,
                pygame.Rect(const.SCREEN_LEFT + 10, 10, 20, 20),
            )

        with mock.patch(
            "src.Battle.battle_draw._render_world_to_surface",
            side_effect=fake_render,
        ):
            BattleDrawController().draw(
                screen,
                [],
                BattleDrawLayout(
                    arena_rect=arena_rect,
                    player1_hud_rect=None,
                    player2_hud_rect=None,
                ),
                (255, 255, 255),
                mock.Mock(),
                options=BattleDrawOptions(draw_huds=False),
            )

        self.assertEqual(screen.get_at((arena_rect.left + 10, 10))[:3], marker)
        self.assertEqual(screen.get_at((const.SCREEN_LEFT + 10, 10))[:3], unchanged)

    def test_hud_panel_is_clipped_to_supplied_rect(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        unchanged = (13, 14, 15)
        screen.fill(unchanged)
        hud_rect = pygame.Rect(40, 50, 320, 120)
        layout = BattleDrawLayout(
            arena_rect=pygame.Rect(0, 0, 1, 1),
            player1_hud_rect=hud_rect,
            player2_hud_rect=None,
        )

        BattleDrawController().draw(
            screen,
            self._empty_snapshot(),
            layout,
            (255, 255, 255),
            mock.Mock(),
            options=BattleDrawOptions(draw_arena=False),
        )

        self.assertNotEqual(
            screen.get_at((hud_rect.left + 1, hud_rect.top + 1))[:3],
            unchanged,
        )
        self.assertEqual(
            screen.get_at((hud_rect.centerx, hud_rect.bottom + 5))[:3],
            unchanged,
        )
        self.assertEqual(
            screen.get_at((hud_rect.right + 5, hud_rect.centery))[:3],
            unchanged,
        )

    def test_live_hud_features_render_into_arbitrary_supplied_rect(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        unchanged = (11, 12, 13)
        screen.fill(unchanged)
        hud_rect = pygame.Rect(410, 300, 320, 240)
        ship = self._ship(1)
        layout = BattleDrawLayout(
            arena_rect=pygame.Rect(0, 0, 1, 1),
            player1_hud_rect=hud_rect,
            player2_hud_rect=None,
        )

        with mock.patch("src.Battle.battle_draw._render_world_to_surface"):
            BattleDrawController().draw(
                screen,
                self._empty_snapshot(ships=(ship,), live_ships=(ship,)),
                layout,
                (255, 255, 255),
                mock.Mock(),
                options=BattleDrawOptions(draw_arena=False),
            )

        hud_content_width = BAR_WIDTH * 2 + VIEWPORT_COLUMN_WIDTH
        draw_x_offset = (hud_rect.width - hud_content_width) // 2
        viewport_left = hud_rect.left + draw_x_offset + BAR_WIDTH + VIEWPORT_MARGIN
        viewport_top = hud_rect.top + MARINE_REGION_HEIGHT
        hp_x = hud_rect.left + draw_x_offset + 2
        hp_y = hud_rect.top + MARINE_REGION_HEIGHT + VIEWPORT_SIZE - 2

        self.assertEqual(
            screen.get_at((viewport_left, viewport_top))[:3],
            const.HUD_VIEWPORT_BORDER,
        )
        self.assertNotEqual(screen.get_at((hp_x, hp_y))[:3], unchanged)
        self.assertEqual(
            screen.get_at((hud_rect.left - 1, hud_rect.top + 1))[:3],
            unchanged,
        )

    def test_draw_battle_wrapper_flips_once(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        with (
            mock.patch("src.Battle.battle_draw.BattleDrawController.draw"),
            mock.patch("src.Battle.battle_draw.pygame.display.flip") as flip,
        ):
            draw_battle(
                screen,
                [],
                pygame.Rect(
                    const.SCREEN_LEFT,
                    0,
                    const.SCREEN_HEIGHT,
                    const.SCREEN_HEIGHT,
                ),
                (255, 255, 255),
                mock.Mock(),
            )

        flip.assert_called_once_with()

    def test_draw_battle_arena_wrapper_delegates_without_flipping(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        arena_rect = pygame.Rect(
            const.SCREEN_LEFT,
            0,
            const.SCREEN_HEIGHT,
            const.SCREEN_HEIGHT,
        )
        controller = mock.Mock()

        with (
            mock.patch(
                "src.Battle.battle_draw.BattleDrawController",
                return_value=controller,
            ),
            mock.patch("src.Battle.battle_draw.pygame.display.flip") as flip,
        ):
            draw_battle_arena(
                screen,
                [],
                arena_rect,
                (255, 255, 255),
                mock.Mock(),
                interp_t=0.25,
            )

        args, kwargs = controller.draw.call_args
        layout = args[2]
        options = kwargs["options"]
        self.assertEqual(layout.arena_rect, arena_rect)
        self.assertIsNone(layout.player1_hud_rect)
        self.assertIsNone(layout.player2_hud_rect)
        self.assertFalse(options.draw_huds)
        self.assertEqual(options.interp_t, 0.25)
        flip.assert_not_called()

    def test_panels_are_at_top_and_control_hints_use_30px_font(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        rendered_text = []

        class RecordingFont:
            def render(self, text, antialias, color):
                rendered_text.append(text)
                return pygame.Surface((180, 24), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Battle.battle_draw._render_world_to_surface"
            ),
            mock.patch(
                "src.Battle.battle_draw.pygame.font.SysFont",
                return_value=RecordingFont(),
            ) as sys_font,
            mock.patch("src.Battle.battle_draw.pygame.display.flip"),
        ):
            draw_battle(
                screen,
                [],
                pygame.Rect(
                    const.SCREEN_LEFT,
                    0,
                    const.SCREEN_HEIGHT,
                    const.SCREEN_HEIGHT,
                ),
                (255, 255, 255),
                mock.Mock(),
            )

        self.assertNotEqual(screen.get_at((285, 100))[:3], (0, 0, 0))
        self.assertEqual(screen.get_at((285, 500))[:3], (0, 0, 0))
        sys_font.assert_called_once_with(None, 30)
        self.assertEqual(rendered_text, ["Press F1 to Pause", "Press Esc to Exit"])

    def test_ai_label_renders_below_player_hud(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        rendered_text = []
        hud_rect = pygame.Rect(40, 50, 320, 320)
        layout = BattleDrawLayout(
            arena_rect=pygame.Rect(0, 0, 1, 1),
            player1_hud_rect=hud_rect,
            player2_hud_rect=None,
        )

        class RecordingFont:
            def render(self, text, antialias, color):
                rendered_text.append(text)
                return pygame.Surface((180, 24), pygame.SRCALPHA)

        with mock.patch(
            "src.Battle.battle_draw.pygame.font.SysFont",
            return_value=RecordingFont(),
        ) as sys_font:
            BattleDrawController().draw(
                screen,
                self._empty_snapshot(),
                layout,
                (255, 255, 255),
                mock.Mock(),
                options=BattleDrawOptions(
                    draw_arena=False,
                    ai_labels={1: "None found"},
                ),
            )

        sys_font.assert_called_once_with(None, HUD_AI_LABEL_FONT_SIZE)
        self.assertEqual(rendered_text, ["AI: None found"])


class BattleResumeFlowTests(unittest.TestCase):
    def test_ai_labels_are_passed_to_battle_draw(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        simulation = mock.Mock()
        simulation.world = []
        simulation.border_rect = pygame.Rect(0, 0, 10, 10)
        simulation.border_color = (255, 255, 255)
        simulation.player1 = mock.Mock()
        simulation.player2 = mock.Mock()
        simulation.entry = None
        simulation.aftermath = None
        simulation.frame_id = 0
        simulation.key_states = {}
        simulation.running = True
        simulation.state.return_value = {"needs_selection": False}
        simulation.step.return_value = {"needs_selection": False}
        fake_clock = SimpleNamespace(
            tick=mock.Mock(side_effect=(1 / const.FPS, 1 / const.FPS)),
            reset=mock.Mock(),
        )
        manager = mock.Mock()
        manager.actions_for_frame.return_value = {}
        manager.label_for_player.side_effect = lambda player: {
            1: "Earthling-01",
            2: "None found",
        }[player]
        escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)

        with (
            mock.patch("src.Battle.battle.BattleSimulation", return_value=simulation),
            mock.patch("src.Battle.battle.BattleAIManager", return_value=manager),
            mock.patch("src.Battle.battle.reset_ai_player_inputs"),
            mock.patch("src.Battle.battle.PresentationClock", return_value=fake_clock),
            mock.patch("src.Battle.battle.confirm_end_match", return_value=True),
            mock.patch(
                "src.Battle.battle.pygame.event.get",
                side_effect=([], [escape]),
            ),
            mock.patch("src.Battle.battle.pygame.event.clear"),
            mock.patch("src.Battle.battle.draw_battle") as draw,
        ):
            battle.run(screen, mock.Mock(), mock.Mock(), player1_ai=True, player2_ai=True)

        self.assertEqual(draw.call_args.kwargs["ai_labels"], {
            1: "Earthling-01",
            2: "None found",
        })

    def test_canceling_escape_does_not_step_or_redraw_before_countdown(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        simulation = mock.Mock()
        simulation.world = []
        simulation.border_rect = pygame.Rect(0, 0, 10, 10)
        simulation.border_color = (255, 255, 255)
        simulation.player1 = mock.Mock()
        simulation.player2 = mock.Mock()
        simulation.entry = None
        simulation.aftermath = None
        simulation.frame_id = 0
        simulation.key_states = {}
        simulation.running = True
        simulation.state.return_value = {}
        fake_clock = SimpleNamespace(
            tick=mock.Mock(return_value=1 / const.FPS),
            reset=mock.Mock(),
        )
        escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)

        with (
            mock.patch("src.Battle.battle.BattleSimulation", return_value=simulation),
            mock.patch("src.Battle.battle.PresentationClock", return_value=fake_clock),
            mock.patch(
                "src.Battle.battle.confirm_end_match", side_effect=(False, True)
            ),
            mock.patch(
                "src.Battle.battle.pygame.event.get",
                side_effect=([escape], [escape]),
            ),
            mock.patch("src.Battle.battle.pygame.event.clear"),
            mock.patch("src.Battle.battle.draw_battle") as draw,
            mock.patch("src.Menus.pick_ship.show_battle_countdown") as countdown,
        ):
            battle.run(screen, mock.Mock(), mock.Mock())

        simulation.step.assert_not_called()
        draw.assert_not_called()
        countdown.assert_called_once()
