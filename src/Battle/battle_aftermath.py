import math
import random
from dataclasses import dataclass, field

import src.const as const
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.Objects.Ships.space_ship import SpaceShip
from src.training import event_ledger
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
class PendingRebirth:
    ship: SpaceShip
    ready_frame: int | None = None


@dataclass
class AftermathState:
    started_frame: int
    latest_death_frame: int
    dead_players: set[int] = field(default_factory=set)
    death_effects: dict[int, list[BattleEffect]] = field(default_factory=dict)
    pending_explosions: list[ScheduledExplosion] = field(default_factory=list)
    ships_pending_hide: set[SpaceShip] = field(default_factory=set)
    camera_hold_targets: list[SpaceShip] = field(default_factory=list)
    victory_ditty_played: bool = False
    initial_victor: SpaceShip | None = None
    victory_cause: object | None = None
    shofixti_won_by_a2: bool = False
    initial_victor_died_permanently: bool = False
    victory_notified: bool = False
    tie_break_ship: SpaceShip | None = None
    choose_second_player: int | None = None
    pending_rebirths: dict[SpaceShip, PendingRebirth] = field(default_factory=dict)
    death_sequence_ready_frame: int | None = None
    conclusion_started_frame: int | None = None
    selection_ready_frame: int | None = None

    @property
    def ditty_started(self):
        """Compatibility name for the monotonic victory-ditty event."""
        return self.victory_ditty_played

    def register_rebirths(self, ships):
        for ship in ships:
            self.pending_rebirths[ship] = PendingRebirth(ship)

    def mark_rebirth_ready_after(self, ship, frame):
        pending = self.pending_rebirths.get(ship)
        if pending is not None:
            pending.ready_frame = frame + const.PKUNK_REBIRTH_PAUSE_FRAMES

    def rebirths_ready(self, frame):
        return [
            pending.ship
            for pending in self.pending_rebirths.values()
            if pending.ready_frame is not None and frame >= pending.ready_frame
        ]

    def finish_rebirths(self, ships):
        for ship in ships:
            self.pending_rebirths.pop(ship, None)


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
    rebirth_ships=(),
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
    state.register_rebirths(rebirth_ships)
    permanent_deaths = [ship for ship in dead_ships if ship not in rebirth_ships]
    if state.initial_victor in permanent_deaths:
        state.initial_victor_died_permanently = True

    for ship in dead_ships:
        ship.current_hp = 0
        ship.currently_alive = False
        ship.reset_controls()
        if hasattr(ship, "position") and hasattr(ship, "previous_position"):
            ship.previous_position = ship.position.copy()
        state.dead_players.add(ship.player)
        with use_audio_service(audio):
            BattleEffect.play_ship_death()
        state.death_effects[ship.player] = []
        explosion_schedule = create_ship_explosion_schedule(ship, frame_id, rng)
        state.pending_explosions.extend(explosion_schedule)
        sequence_ready_frame = (
            explosion_schedule[-1].frame + const.POST_DEATH_EFFECT_FRAMES
        )
        state.death_sequence_ready_frame = max(
            state.death_sequence_ready_frame or sequence_ready_frame,
            sequence_ready_frame,
        )
        state.ships_pending_hide.add(ship)
        state.latest_death_frame = frame_id

    # Camera framing follows the current death event. Retaining older dead ships
    # here causes a late survivor death to zoom out to both wreck positions.
    state.camera_hold_targets = list(dead_ships)

    release_dead_opponents(game_objects, dead_ships)

    record_resolved_victory(
        state,
        player1,
        player2,
        permanent_deaths=permanent_deaths,
    )

    if state.initial_victor_died_permanently:
        state.selection_ready_frame = state.death_sequence_ready_frame

    audio.stop_music()
    return state


def record_resolved_victory(
    aftermath: AftermathState,
    player1,
    player2,
    *,
    permanent_deaths=(),
):
    """Record the first resolved victory without deriving it from later state."""
    if aftermath.initial_victor is not None or aftermath.pending_rebirths:
        return aftermath.initial_victor

    permanent_deaths = list(permanent_deaths)
    if len(permanent_deaths) == 2:
        victor, cause = _shofixti_a2_victory(permanent_deaths)
        if victor is None:
            return None
        aftermath.initial_victor = victor
        aftermath.victory_cause = cause
        aftermath.shofixti_won_by_a2 = True
        aftermath.tie_break_ship = victor
        aftermath.choose_second_player = victor.player
        return victor

    living_ships = [
        ship
        for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    if len(living_ships) != 1:
        return None

    victor = living_ships[0]
    aftermath.initial_victor = victor
    if len(permanent_deaths) == 1:
        aftermath.victory_cause = getattr(
            permanent_deaths[0],
            "last_lethal_damage_source",
            None,
        )
    return victor


def _shofixti_a2_victory(permanent_deaths):
    for candidate in permanent_deaths:
        if not getattr(candidate, "shofixti_self_destruct", False):
            continue
        opponent = next(
            (ship for ship in permanent_deaths if ship is not candidate),
            None,
        )
        cause = getattr(opponent, "last_lethal_damage_source", None)
        if (
            getattr(cause, "name", None) == "ShofixtiA2"
            and getattr(cause, "parent", None) is candidate
        ):
            return candidate, cause
    return None, None


def release_dead_opponents(game_objects, dead_ships):
    world = World.coerce(game_objects)
    dead_ids = {id(ship) for ship in dead_ships}
    for obj in world:
        opponent = getattr(obj, "opponent", None)
        if opponent is None or id(opponent) not in dead_ids:
            continue
        on_opponent_lost = getattr(obj, "on_opponent_lost", None)
        if on_opponent_lost is not None:
            on_opponent_lost(opponent)
        else:
            obj.opponent = None


def restore_reborn_opponents(game_objects, reborn_ships):
    """Reconnect surviving abilities to ships that rejoined the same round."""
    world = World.coerce(game_objects)
    reborn_ships = tuple(reborn_ships)
    tracked_ability_types = {"projectile", "laser", "special_object"}

    for obj in world:
        if (
            getattr(obj, "type", None) not in tracked_ability_types
            or not World.is_alive(obj)
            or getattr(obj, "opponent", None) is not None
        ):
            continue

        opponent = next(
            (ship for ship in reborn_ships if ship.player != obj.player),
            None,
        )
        if opponent is None:
            continue

        on_opponent_restored = getattr(obj, "on_opponent_restored", None)
        if on_opponent_restored is not None:
            on_opponent_restored(opponent)
        else:
            obj.opponent = opponent


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
        base_pos = getattr(ship, "previous_position", ship.position)
        position = [
            (base_pos[0] + local_x * cos_a - local_y * sin_a) % const.ARENA_SIZE,
            (base_pos[1] + local_x * sin_a + local_y * cos_a) % const.ARENA_SIZE,
        ]
        schedule.append(
            ScheduledExplosion(
                frame=start_frame + index * EXPLOSION_PLACEMENT_INTERVAL_FRAMES,
                ship=ship,
                position=position,
                scale=rng.uniform(0.85, 1.15),
                is_final=index == count - 1,
            )
        )

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
        item for item in aftermath.pending_explosions if item.frame <= frame_id
    ]
    aftermath.pending_explosions = [
        item for item in aftermath.pending_explosions if item.frame > frame_id
    ]

    for item in ready_explosions:
        effect = BattleEffect.ship_explosion(item.position, scale=item.scale)
        aftermath.death_effects[item.ship.player].append(effect)
        world.add(effect)
        if item.is_final:
            with use_audio_service(audio):
                BattleEffect.play_ship_death()
            aftermath.mark_rebirth_ready_after(item.ship, frame_id)
            hide_dead_ship(item.ship, world)
            aftermath.ships_pending_hide.discard(item.ship)

    death_view_done = (
        aftermath.death_sequence_ready_frame is not None
        and frame_id >= aftermath.death_sequence_ready_frame
    )
    if aftermath.pending_rebirths:
        return
    if not death_view_done:
        return

    record_resolved_victory(aftermath, player1, player2)

    if aftermath.initial_victor_died_permanently:
        aftermath.conclusion_started_frame = (
            aftermath.conclusion_started_frame or frame_id
        )
        aftermath.selection_ready_frame = frame_id
        return

    if aftermath.victory_ditty_played:
        return

    victory_ship = aftermath.initial_victor
    if victory_ship is not None:
        if audio_service is None and sound_enabled:
            play_victory_ditty(victory_ship)
        else:
            audio.play_victory_ditty(victory_ship)
        aftermath.victory_ditty_played = True
        aftermath.selection_ready_frame = frame_id + const.VICTORY_DITTY_VIEW_FRAMES
    else:
        aftermath.selection_ready_frame = frame_id
    aftermath.conclusion_started_frame = frame_id


def hide_dead_ship(ship, game_objects):
    world = World.coerce(game_objects)
    removed_owners = [ship]
    pending = world.abilities
    while pending:
        removed_any = False
        for ability in pending[:]:
            if not any(
                getattr(ability, "parent", None) is owner
                for owner in removed_owners
            ):
                continue
            ability.on_parent_removed()
            if not ability.is_alive():
                event_ledger.record_removed(
                    ability,
                    destroyed=False,
                    reason="parent_cleanup",
                )
                removed_owners.append(ability)
            pending.remove(ability)
            removed_any = True
        if not removed_any:
            break
    world.remove_dead_collision_objects()
    world.remove_where(lambda obj: obj is ship)
    if hasattr(ship, "position") and hasattr(ship, "previous_position"):
        ship.previous_position = ship.position.copy()


def aftermath_camera_targets(
    aftermath: AftermathState | None,
    player1,
    player2,
    frame_id=None,
):
    if aftermath is None:
        return None

    targets = [
        ship
        for ship in (player1, player2)
        if ship.currently_alive and ship.current_hp > 0
    ]
    if aftermath.pending_rebirths:
        targets.extend(
            ship
            for ship in aftermath.camera_hold_targets
            if ship in aftermath.pending_rebirths
        )
        return targets or None

    if (
        frame_id is not None
        and aftermath.death_sequence_ready_frame is not None
        and frame_id >= aftermath.death_sequence_ready_frame
    ):
        if (
            aftermath.initial_victor is not None
            and not aftermath.initial_victor_died_permanently
        ):
            return [aftermath.initial_victor]
        return None

    targets.extend(aftermath.camera_hold_targets)
    return targets or None


def aftermath_ready_for_selection(
    aftermath: AftermathState,
    frame_id,
    sound_enabled=True,
):
    if aftermath.pending_rebirths or aftermath.conclusion_started_frame is None:
        return False
    if aftermath.selection_ready_frame is not None:
        return frame_id >= aftermath.selection_ready_frame
    elapsed = frame_id - aftermath.conclusion_started_frame
    return elapsed >= const.VICTORY_DITTY_VIEW_FRAMES


def play_victory_ditty(ship):
    PygameAudioService(getattr(ship, "resources", default_assets())).play_victory_ditty(
        ship
    )
