"""Authoritative battle object storage and typed object queries."""

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from src.Battle.effects import BattleEffect
from src.Objects.object import Object, ThrustMarker
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip

T = TypeVar("T")


class World:
    """Owns battle objects in their stable simulation order.

    Typed properties are snapshots derived from the authoritative list. No
    secondary collection is retained, so additions and removals cannot leave
    an index out of sync.
    """

    def __init__(self, objects: Iterable[Any] | None = None):
        if isinstance(objects, World):
            self._objects = objects.objects
        elif isinstance(objects, list):
            self._objects = objects
        else:
            self._objects = list(objects or [])

    @classmethod
    def coerce(cls, objects: "World | list[Any]") -> "World":
        return objects if isinstance(objects, cls) else cls(objects)

    @property
    def objects(self) -> list[Any]:
        """The compatibility list, also used as the sole source of ordering."""
        return self._objects

    def __iter__(self) -> Iterator[Any]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, obj: Any) -> bool:
        return obj in self._objects

    def snapshot(self) -> list[Any]:
        return self._objects[:]

    def add(self, obj: Any) -> None:
        if hasattr(obj, "position") and hasattr(obj, "previous_position"):
            obj.previous_position = obj.position.copy()
        self._objects.append(obj)

    def add_all(self, objects: Iterable[Any]) -> None:
        objects_list = list(objects)
        for obj in objects_list:
            if hasattr(obj, "position") and hasattr(obj, "previous_position"):
                obj.previous_position = obj.position.copy()
        self._objects.extend(objects_list)

    def remove(self, obj: Any) -> None:
        self._objects.remove(obj)

    def remove_where(self, predicate) -> None:
        self._objects[:] = [obj for obj in self._objects if not predicate(obj)]

    def retain(self, objects: Iterable[Any]) -> None:
        self._objects[:] = list(objects)

    def objects_of_type(self, object_type: type[T]) -> list[T]:
        return [obj for obj in self._objects if isinstance(obj, object_type)]

    def objects_of_types(self, *object_types: type[T]) -> list[T]:
        return [obj for obj in self._objects if isinstance(obj, object_types)]

    def objects_excluding_types(self, *object_types: type) -> list[Any]:
        return [obj for obj in self._objects if not isinstance(obj, object_types)]

    @property
    def ships(self) -> list[SpaceShip]:
        return self.objects_of_type(SpaceShip)

    @property
    def abilities(self) -> list[Ability]:
        return self.objects_of_type(Ability)

    def abilities_of_kind(self, *kinds: str) -> list[Ability]:
        return [ability for ability in self.abilities if ability.type in kinds]

    @property
    def projectiles(self) -> list[Ability]:
        return self.abilities_of_kind("projectile")

    @property
    def special_objects(self) -> list[Ability]:
        return self.abilities_of_kind("special_object")

    @property
    def lasers(self) -> list[Ability]:
        return self.abilities_of_kind("laser")

    @property
    def area_abilities(self) -> list[Ability]:
        return self.abilities_of_kind("area")

    @property
    def asteroids(self) -> list[Asteroid]:
        return self.objects_of_type(Asteroid)

    @property
    def planets(self) -> list[Planet]:
        return self.objects_of_type(Planet)

    @property
    def stars(self) -> list[Star]:
        return self.objects_of_type(Star)

    @property
    def effects(self) -> list[BattleEffect]:
        return self.objects_of_type(BattleEffect)

    @property
    def thrust_markers(self) -> list[ThrustMarker]:
        return self.objects_of_type(ThrustMarker)

    @staticmethod
    def is_alive(obj: Any) -> bool:
        if isinstance(obj, Object):
            return obj.is_alive()

        # Compatibility boundary for lightweight test and integration doubles.
        return (
            getattr(obj, "currently_alive", True) and getattr(obj, "current_hp", 1) > 0
        )

    @staticmethod
    def participates_in_collision(obj: Any) -> bool:
        if isinstance(obj, Object):
            return obj.can_collide and World.is_alive(obj)
        return getattr(obj, "can_collide", False) and World.is_alive(obj)

    @staticmethod
    def is_colliding_ability_kind(obj: Any, kind: str) -> bool:
        return (
            isinstance(obj, Ability)
            and obj.type == kind
            and World.participates_in_collision(obj)
        )

    @property
    def live_ships(self) -> list[SpaceShip]:
        return [ship for ship in self.ships if self.is_alive(ship)]

    def colliding_abilities_of_kind(self, kind: str) -> list[Ability]:
        return [
            ability
            for ability in self.abilities_of_kind(kind)
            if self.is_colliding_ability_kind(ability, kind)
        ]

    @property
    def colliding_projectiles(self) -> list[Ability]:
        return self.colliding_abilities_of_kind("projectile")

    @property
    def colliding_special_objects(self) -> list[Ability]:
        return self.colliding_abilities_of_kind("special_object")

    @property
    def colliding_lasers(self) -> list[Ability]:
        return self.colliding_abilities_of_kind("laser")

    @property
    def live_asteroids(self) -> list[Asteroid]:
        return [asteroid for asteroid in self.asteroids if asteroid.currently_alive]

    @property
    def pending_area_damage(self) -> list[Any]:
        return [
            obj
            for obj in self._objects
            if (
                obj.area_damage_capabilities.emits
                and obj.currently_alive
                and obj.area_damage_pending
            )
        ]

    @property
    def asteroid_spawn_avoid_bodies(self) -> list[Any]:
        avoid_bodies = []
        for obj in self._objects:
            if isinstance(obj, Planet):
                continue
            if isinstance(obj, Asteroid):
                if obj.currently_alive:
                    avoid_bodies.append(obj)
                continue
            if isinstance(obj, SpaceShip):
                if obj.current_hp > 0:
                    avoid_bodies.append(obj)
                continue
            if isinstance(obj, Ability):
                if self.participates_in_collision(obj):
                    avoid_bodies.append(obj)
                continue
            if isinstance(obj, Object):
                if obj.can_collide and obj.is_alive():
                    avoid_bodies.append(obj)
                continue
            if getattr(obj, "can_collide", False) and getattr(
                obj, "currently_alive", True
            ):
                avoid_bodies.append(obj)
        return avoid_bodies

    def update_objects(self, excluded_objects=()) -> None:
        """Update one stable snapshot and append drained spawns afterward."""
        excluded_ids = {id(obj) for obj in excluded_objects}
        spawned_objects = []
        for obj in self.snapshot():
            if id(obj) in excluded_ids:
                continue
            if isinstance(obj, SpaceShip) and obj.current_hp <= 0:
                continue
            alive = obj.update()
            spawned_objects.extend(self._drain_spawned_objects(obj))
            if not alive:
                self.remove(obj)
        self.add_all(spawned_objects)

    @staticmethod
    def _drain_spawned_objects(obj: Any) -> Iterable[Any]:
        if isinstance(obj, Object):
            return obj.drain_spawned_objects()

        # Compatibility boundary for lightweight simulation test doubles.
        drain = getattr(obj, "drain_spawned_objects", None)
        return drain() if drain is not None else ()

    def remove_dead_collision_objects(self) -> None:
        self.remove_where(
            lambda obj: (
                isinstance(obj, (Ability, Asteroid)) and not obj.currently_alive
            )
        )
