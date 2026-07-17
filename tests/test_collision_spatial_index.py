import random
import unittest
from types import SimpleNamespace
from unittest import mock

from src.Battle import collisions
from src.Battle.collision_contract import CollisionOutcome
from src.Battle.collision_geometry import objects_overlap_during_frame
from src.Battle.collision_spatial_index import ToroidalSpatialIndex
from src.collision_capabilities import PhysicalCollisionCapabilities
from src import const


def body(position, *, previous=None, size=(20, 20), mask=None):
    obj = SimpleNamespace(
        position=list(position),
        previous_position=list(previous if previous is not None else position),
        size=list(size),
        can_move=True,
        currently_alive=True,
        physical_collision_capabilities=PhysicalCollisionCapabilities(),
    )
    obj.get_collision_mask = lambda: mask
    return obj


def index_for(objects, *, category="objects", cell_size=128):
    categories = {id(obj): (category,) for obj in objects}
    return ToroidalSpatialIndex(
        objects,
        categories=categories,
        cell_size=cell_size,
    )


class ToroidalSpatialIndexTests(unittest.TestCase):
    def test_cross_category_candidate_cache_is_invalidated_by_updates(self):
        first = body((100, 100))
        category_a = body((110, 100))
        category_b = body((120, 100))
        index = ToroidalSpatialIndex(
            [first, category_a, category_b],
            categories={
                id(first): ("first",),
                id(category_a): ("a",),
                id(category_b): ("b",),
            },
            cell_size=64,
        )

        self.assertEqual(index.candidates_for(first, categories=("a",)), [category_a])
        self.assertEqual(index.candidates_for(first, categories=("b",)), [category_b])
        category_b.position = [1000, 1000]
        category_b.previous_position = category_b.position.copy()
        self.assertTrue(index.update(category_b))
        self.assertEqual(index.candidates_for(first, categories=("b",)), [])

    def test_same_and_adjacent_cell_candidates(self):
        first = body((100, 100))
        same_cell = body((115, 100))
        adjacent_cell = body((130, 100))
        distant = body((500, 500))
        index = index_for([first, same_cell, adjacent_cell, distant])

        self.assertEqual(
            index.candidates_for(first, categories=("objects",)),
            [same_cell, adjacent_cell],
        )

    def test_large_multicell_object_is_returned_once(self):
        first = body((500, 500), size=(500, 500))
        second = body((520, 520), size=(500, 500))
        index = index_for([first, second], cell_size=64)

        self.assertEqual(index.candidates_for(first), [second])

    def test_candidate_order_is_world_order_not_cell_or_hash_order(self):
        first = body((200, 200), size=(300, 300))
        earlier = body((290, 200))
        later = body((110, 200))
        index = index_for([first, earlier, later], cell_size=64)

        for _ in range(3):
            self.assertEqual(index.candidates_for(first), [earlier, later])

    def test_wrapped_edge_and_corner_candidates(self):
        edge = body((5, 400))
        across_edge = body((const.ARENA_SIZE - 5, 400))
        corner = body((5, 5))
        across_corner = body((const.ARENA_SIZE - 5, const.ARENA_SIZE - 5))
        index = index_for([edge, across_edge, corner, across_corner])

        self.assertIn(across_edge, index.candidates_for(edge))
        self.assertIn(across_corner, index.candidates_for(corner))

    def test_fast_sweep_crosses_multiple_cells(self):
        moving = body((900, 500), previous=(100, 500))
        target = body((500, 500))
        index = index_for([moving, target], cell_size=100)

        self.assertIn(target, index.candidates_for(moving))

    def test_short_seam_crossing_does_not_cover_unrelated_arena_cells(self):
        moving = body((100, 500), previous=(const.ARENA_SIZE - 100, 500))
        seam_target = body((5, 500))
        middle_target = body((const.ARENA_SIZE / 2, 500))
        index = index_for([moving, seam_target, middle_target], cell_size=100)

        candidates = index.candidates_for(moving)
        self.assertIn(seam_target, candidates)
        self.assertNotIn(middle_target, candidates)

    def test_current_size_change_is_applied_by_update(self):
        changing = body((1000, 1000), size=(20, 20))
        target = body((1190, 1000), size=(20, 20))
        index = index_for([changing, target], cell_size=64)
        self.assertNotIn(target, index.candidates_for(changing))

        changing.size = [400, 400]
        self.assertTrue(index.update(changing))
        self.assertIn(target, index.candidates_for(changing))

    def test_segment_query_wraps_and_suppresses_duplicates(self):
        first = body((const.ARENA_SIZE - 20, 500), size=(60, 60))
        second = body((20, 500), size=(60, 60))
        unrelated = body((4000, 500))
        index = index_for([first, second, unrelated], category="laser_targets")

        candidates = index.query_segments(
            (((const.ARENA_SIZE - 100, 500), (100, 500)),),
            width=12,
            categories=("laser_targets",),
        )

        self.assertEqual(candidates, [first, second])

    def test_radius_query_includes_target_extent(self):
        emitter_position = (1000, 1000)
        large_target = body((1250, 1000), size=(200, 200))
        index = index_for([large_target], category="area_targets", cell_size=64)

        self.assertEqual(
            index.query_radius(
                emitter_position,
                160,
                categories=("area_targets",),
            ),
            [large_target],
        )

    def test_unique_spatial_pairs_are_dispatched_once(self):
        objects = [body((100, 100)), body((110, 100)), body((120, 100))]
        index = index_for(objects, category="asteroids")
        groups = {
            "ships": [],
            "asteroids": objects,
            "projectiles": [],
            "special_objects": [],
            "planets": [],
        }

        with mock.patch.object(
            collisions,
            "_dispatch_collision_pair",
            return_value=CollisionOutcome.IGNORED,
        ) as dispatch:
            collisions._run_collision_phases(
                groups,
                [],
                spatial_index=index,
            )

        self.assertEqual(dispatch.call_count, 3)
        pairs = [tuple(call.args[:2]) for call in dispatch.call_args_list]
        self.assertEqual(
            pairs,
            [
                (objects[0], objects[1]),
                (objects[0], objects[2]),
                (objects[1], objects[2]),
            ],
        )

    def test_outer_object_requeries_after_collision_repositioning(self):
        first = body((100, 100))
        initial_contact = body((105, 100))
        later_contact = body((1000, 100))
        objects = [first, initial_contact, later_contact]
        index = index_for(objects, category="objects", cell_size=64)
        calls = []

        def dispatch(outer, other, context):
            calls.append((outer, other))
            if other is initial_contact:
                outer.position = later_contact.position.copy()
                return CollisionOutcome.RESOLVED
            return CollisionOutcome.IGNORED

        with mock.patch.object(collisions, "_dispatch_collision_pair", dispatch):
            collisions._dispatch_collision_pairs(
                [first],
                [initial_contact, later_contact],
                [],
                stop_after_handled=False,
                spatial_index=index,
                second_category="objects",
            )

        self.assertEqual(calls, [(first, initial_contact), (first, later_contact)])

    def test_randomized_exact_collisions_are_never_omitted(self):
        rng = random.Random(7319)
        for scene_index in range(30):
            objects = []
            for _ in range(18):
                position = [
                    rng.uniform(0, const.ARENA_SIZE),
                    rng.uniform(0, const.ARENA_SIZE),
                ]
                previous = [
                    (position[0] + rng.uniform(-600, 600)) % const.ARENA_SIZE,
                    (position[1] + rng.uniform(-600, 600)) % const.ARENA_SIZE,
                ]
                dimension = rng.uniform(4, 180)
                objects.append(
                    body(position, previous=previous, size=(dimension, dimension))
                )

            index = index_for(objects, cell_size=128)
            for first_index, first in enumerate(objects):
                candidates = {id(obj) for obj in index.candidates_for(first)}
                for second in objects[first_index + 1 :]:
                    if objects_overlap_during_frame(first, second):
                        self.assertIn(
                            id(second),
                            candidates,
                            msg=f"scene {scene_index} omitted an exact collision",
                        )


if __name__ == "__main__":
    unittest.main()
