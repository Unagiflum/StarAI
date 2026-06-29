import unittest
from types import SimpleNamespace
from unittest import mock

from src.Battle import collision_responses, collisions
from src.Battle.collision_contract import CollisionContext, CollisionOutcome
from src.Battle.collision_dispatch import CollisionPairRegistry
from src.collision_capabilities import CollisionCapabilities, CollisionRole


def collision_object(role):
    return SimpleNamespace(
        collision_capabilities=CollisionCapabilities(role),
        physical_collision_capabilities=None,
    )


class CollisionPairRegistryTests(unittest.TestCase):
    def test_migrated_pairs_use_dedicated_mobile_solid_handler(self):
        for first_role, second_role in (
            (CollisionRole.SHIP, CollisionRole.SHIP),
            (CollisionRole.ASTEROID, CollisionRole.ASTEROID),
            (CollisionRole.SHIP, CollisionRole.ASTEROID),
        ):
            with self.subTest(first_role=first_role, second_role=second_role):
                handler = collisions.COLLISION_PAIR_REGISTRY.handler_for(
                    first_role,
                    second_role,
                )

                self.assertIs(
                    handler,
                    collision_responses.resolve_mobile_solid_collision,
                )

    def test_bidirectional_registration_preserves_incoming_object_order(self):
        registry = CollisionPairRegistry()
        handler = mock.Mock(return_value=CollisionOutcome.RESOLVED)
        registry.register(CollisionRole.SHIP, CollisionRole.ASTEROID, handler)
        asteroid = collision_object(CollisionRole.ASTEROID)
        ship = collision_object(CollisionRole.SHIP)
        context = CollisionContext([])

        outcome = registry.dispatch(asteroid, ship, context)

        self.assertIs(outcome, CollisionOutcome.RESOLVED)
        handler.assert_called_once_with(asteroid, ship, context)

    def test_unregistered_pair_is_ignored(self):
        registry = CollisionPairRegistry()
        context = CollisionContext([])

        outcome = registry.dispatch(
            collision_object(CollisionRole.LASER),
            collision_object(CollisionRole.SHIP),
            context,
        )

        self.assertIs(outcome, CollisionOutcome.IGNORED)

    def test_duplicate_registration_is_rejected(self):
        registry = CollisionPairRegistry()
        handler = mock.Mock(return_value=CollisionOutcome.RESOLVED)
        registry.register(CollisionRole.SHIP, CollisionRole.ASTEROID, handler)

        with self.assertRaisesRegex(ValueError, "ASTEROID x SHIP"):
            registry.register(CollisionRole.ASTEROID, CollisionRole.SHIP, handler)

    def test_failed_bidirectional_registration_is_atomic(self):
        registry = CollisionPairRegistry()
        existing = mock.Mock(return_value=CollisionOutcome.RESOLVED)
        replacement = mock.Mock(return_value=CollisionOutcome.RESOLVED)
        registry.register(
            CollisionRole.ASTEROID,
            CollisionRole.SHIP,
            existing,
            bidirectional=False,
        )

        with self.assertRaisesRegex(ValueError, "ASTEROID x SHIP"):
            registry.register(
                CollisionRole.SHIP,
                CollisionRole.ASTEROID,
                replacement,
            )

        self.assertIsNone(
            registry.handler_for(CollisionRole.SHIP, CollisionRole.ASTEROID)
        )
        self.assertIs(
            registry.handler_for(CollisionRole.ASTEROID, CollisionRole.SHIP),
            existing,
        )

    def test_collision_pipeline_dispatches_registered_role_pair(self):
        first = collision_object(CollisionRole.SHIP)
        second = collision_object(CollisionRole.ASTEROID)
        context = CollisionContext([])
        registry = mock.Mock()
        registry.dispatch.return_value = CollisionOutcome.RESOLVED

        with mock.patch.object(collisions, "COLLISION_PAIR_REGISTRY", registry):
            outcome = collisions._dispatch_collision_pair(first, second, context)

        self.assertIs(outcome, CollisionOutcome.RESOLVED)
        registry.dispatch.assert_called_once_with(first, second, context)


if __name__ == "__main__":
    unittest.main()
