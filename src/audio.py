"""Narrow audio boundary used by gameplay and headless simulations."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol

import src.const as const


class AudioService(Protocol):
    """Audio operations required by StarAI gameplay."""

    enabled: bool

    def start_battle_music(self) -> None: ...
    def stop_music(self) -> None: ...
    def play_victory_ditty(self, ship) -> None: ...
    def play_effect(self, path: Path, volume: float = const.SOUND_EFFECT_VOLUME) -> float: ...
    def load_effect(self, path: Path, volume: float = const.SOUND_EFFECT_VOLUME): ...


class PygameAudioService:
    """Pygame adapter. AssetManager remains the owner of the sound cache."""

    def __init__(self, resources=None, enabled=True):
        self.enabled = bool(enabled)
        if resources is None:
            from src.resources import default_assets
            resources = default_assets()
        self.resources = resources

    def start_battle_music(self):
        if self.enabled:
            self.resources.play_music(
                const.BATTLE_MUSIC_PATH, const.BATTLE_MUSIC_VOLUME, loops=-1
            )

    def stop_music(self):
        if self.enabled:
            import pygame
            pygame.mixer.music.stop()

    def play_victory_ditty(self, ship):
        if not self.enabled:
            return
        import pygame
        try:
            resources = getattr(ship, "resources", self.resources)
            resources.play_music(
                resources.ship(ship.name).ditty_path,
                const.BATTLE_MUSIC_VOLUME,
            )
        except pygame.error:
            pass

    def load_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        if not self.enabled:
            return None
        return self.resources.sound(path, volume, enabled=True)

    def play_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        sound = self.load_effect(path, volume)
        if not sound:
            return 0.0
        sound.play()
        return sound.get_length()


class NullAudioService:
    """No-op adapter for disabled and headless execution."""

    enabled = False

    def start_battle_music(self):
        pass

    def stop_music(self):
        pass

    def play_victory_ditty(self, ship):
        pass

    def load_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        return None

    def play_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        return 0.0


class RecordingAudioService(NullAudioService):
    """Pygame-free test adapter recording operation order and arguments."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.operations = []

    def _record(self, name, *args):
        if self.enabled:
            self.operations.append((name, *args))

    def start_battle_music(self):
        self._record("start_battle_music")

    def stop_music(self):
        self._record("stop_music")

    def play_victory_ditty(self, ship):
        self._record("play_victory_ditty", ship)

    def play_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        self._record("play_effect", Path(path), volume)
        return 0.0

    def load_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        if not self.enabled:
            return None
        return _RecordingEffect(self, Path(path), volume)


class _RecordingEffect:
    def __init__(self, service, path, volume):
        self.service = service
        self.path = path
        self.volume = volume

    def play(self):
        self.service.play_effect(self.path, self.volume)

    def get_length(self):
        return 0.0


_active_audio = ContextVar("starai_active_audio", default=None)


@contextmanager
def use_audio_service(service):
    """Scope legacy collision-effect calls to one simulation instance."""
    token = _active_audio.set(service)
    try:
        yield
    finally:
        _active_audio.reset(token)


def active_audio_service():
    return _active_audio.get()


def compatibility_audio_service(enabled=True, resources=None):
    """Construct the legacy default without mutating global enablement state."""
    if not enabled:
        return NullAudioService()
    return PygameAudioService(resources=resources)
