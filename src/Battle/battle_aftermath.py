import math
import random
from dataclasses import dataclass, field

import src.const as const
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.Objects.Ships.space_ship import SpaceShip
from src.audio import (
    PygameAudioService,
    compatibility_audio_service,
    use_audio_service,
)
from src.resources import default_assets


EXPLOSION_PLACEMENT_INTERVAL_FRAMES = 3


@dataclass
class ScheduledExplosion:
    frame: int
    ship: SpaceShip
    position: list[float]
    scale: float
    is_final: bool


@dataclass
class AftermathState:
    started_frame: int
    latest_death_frame: int
    dead_players: set[int] = field(default_factory=set)
    death_effects: dict[int, list[BattleEffect]] = field(default_factory=dict)
    pending_explosions: list[ScheduledExplosion] = field(default_factory=list)
    ships_pending_hide: set[SpaceShip] = field(default_factory=set)
    camera_hold_targets: list[SpaceShip] = field(default_factory=list)
    ditty_started: bool = False
    tie_break_ship: SpaceShip | None = None
    choose_second_player: int | None = None


def start_or_update_aftermath(
    aftermath: AftermathState | None,
    dead_ships,
    player1,
    player2,
    game_objects,
    frame_id,
    sound_enabled=True,
    audio_service=None,
    rng=None,
) -> AftermathState:
    rng = rng or random
    audio = (
        audio_service
        if audio_service is not None
        else compatibility_audio_service(sound_enabled)
    )
    state = aftermath or AftermathState(
        started_frame=frame_id,
        latest_death_frame=frame_id,
    )

    for ship in dead_ships:
        ship.current_hp = 0
        ship.currently_alive = False
        ship.reset_controls()
        state.dead_players.add(ship.player)
        with use_audio_service(audio):
            BattleEffect.play_ship_death()
        state.death_effects[ship.player] = []
        state.pending_explosions.extend(
            create_ship_explosion_schedule(ship, frame_id, rng)
        )
        state.ships_pending_hide.add(ship)
        state.camera_hold_targets.append(ship)
        state.latest_death_frame = frame_id

    if player1.current_hp <= 0 and player2.current_hp <= 0:
        self_destructors = [
            ship for ship in (player1, player2)
            if getattr(ship, "shofixti_self_destruct", False)
        ]
        if len(self_destructors) == 1:
            state.tie_break_ship = self_destructors[0]
            state.choose_second_player = self_destructors[0].player

    audio.stop_music()
    state.ditty_started = False
    return state


def create_ship_explosion_schedule(ship, start_frame, rng=None):
    rng = rng or random
    count = max(4, min(9, int(max(ship.size) / 35) + 3))
    schedule = []
    angle = math.radians(ship.rotation)
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)

    for index in range(count):
        local_x = rng.uniform(-ship.size[0] * 0.45, ship.size[0] * 0.45)
        local_y = rng.uniform(-ship.size[1] * 0.45, ship.size[1] * 0.45)
        position = [
            (ship.position[0] + local_x * cos_a - local_y * sin_a) % const.ARENA_SIZE,
            (ship.position[1] + local_x * sin_a + local_y * cos_a) % const.ARENA_SIZE,
        ]
        schedule.append(ScheduledExplosion(
            frame=start_frame + index * EXPLOSION_PLACEMENT_INTERVAL_FRAMES,
            ship=ship,
            position=position,
            scale=rng.uniform(0.85, 1.15),
            is_final=index == count - 1,
        ))

    return schedule


def update_aftermath(
    aftermath: AftermathState,
    player1,
    player2,
    game_objects,
    frame_id,
    sound_enabled=True,
    audio_service=None,
):
    audio = (
        audio_service
        if audio_service is not None
        else compatibility_audio_service(sound_enabled)
    )
    world = World.coerce(game_objects)
    ready_explosions = [
        item for item in aftermath.pending_explosions
        if item.frame <= frame_id
    ]
    aftermath.pending_explosions = [
        item for item in aftermath.pending_explosions
        if item.frame > frame_id
    ]

    for item in ready_explosions:
        effect = BattleEffect.ship_explosion(item.position, scale=item.scale)
        aftermath.death_effects[item.ship.player].append(effect)
        world.add(effect)
        if item.is_final:
            with use_audio_service(audio):
                BattleEffect.play_ship_death()
            hide_dead_ship(item.ship, world)
            aftermath.ships_pending_hide.discard(item.ship)

    living_ships = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    death_view_done = (
        frame_id - aftermath.started_frame
        >= const.POST_DEATH_ANIMATION_VIEW_FRAMES
    )
    if len(living_ships) == 1 and not aftermath.ditty_started and death_view_done:
        if audio_service is None and sound_enabled:
            play_victory_ditty(living_ships[0])
        else:
            audio.play_victory_ditty(living_ships[0])
        aftermath.ditty_started = True
    elif (
        not living_ships
        and aftermath.tie_break_ship is not None
        and not aftermath.ditty_started
        and death_view_done
    ):
        if audio_service is None and sound_enabled:
            play_victory_ditty(aftermath.tie_break_ship)
        else:
            audio.play_victory_ditty(aftermath.tie_break_ship)
        aftermath.ditty_started = True


def hide_dead_ship(ship, game_objects):
    World.coerce(game_objects).remove_where(lambda obj: obj is ship)


def aftermath_camera_targets(
    aftermath: AftermathState | None,
    player1,
    player2,
    frame_id=None,
):
    if aftermath is None:
        return None
    if (
        frame_id is not None
        and frame_id - aftermath.started_frame
        >= const.POST_DEATH_ANIMATION_VIEW_FRAMES
    ):
        return None

    targets = [
        ship for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    targets.extend(aftermath.camera_hold_targets)
    return targets or None


def aftermath_ready_for_selection(
    aftermath: AftermathState,
    frame_id,
    sound_enabled=True,
):
    elapsed = frame_id - aftermath.started_frame
    return elapsed >= const.POST_DEATH_CONTROL_FRAMES


def play_victory_ditty(ship):
    PygameAudioService(
        getattr(ship, "resources", default_assets())
    ).play_victory_ditty(ship)
