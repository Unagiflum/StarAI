from unittest import mock

from collision_test_support import CollisionTestCase
from src.Battle import collision_responses
from src.Battle.collision_contract import (
    CollisionContext,
    CollisionEnvironment,
    CollisionOutcome,
    collision_context,
)


class CollisionOutcomeTests(CollisionTestCase):
    def make_parent_contact(self, *, hit_parent):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        parent.previous_position = parent.position.copy()
        parent.take_damage = mock.Mock(return_value=1)

        projectile = self.make_projectile(parent)
        projectile.player = 1
        projectile.hit_parent = hit_parent
        projectile.position = parent.position.copy()
        projectile.previous_position = projectile.position.copy()
        projectile.on_ship_impact = mock.Mock()
        return parent, projectile

    def test_ignored_pair_returns_explicit_outcome(self):
        parent, projectile = self.make_parent_contact(hit_parent=False)

        outcome = collision_responses.resolve_generic_collision(
            projectile,
            parent,
            CollisionContext([]),
        )

        self.assertIs(outcome, CollisionOutcome.IGNORED)
        self.assertFalse(outcome)

    def test_resolved_pair_returns_explicit_outcome(self):
        parent, projectile = self.make_parent_contact(hit_parent=True)

        with mock.patch.object(collision_responses.BattleEffect, "from_blast"):
            outcome = collision_responses.resolve_generic_collision(
                projectile,
                parent,
                CollisionContext([]),
            )

        self.assertIs(outcome, CollisionOutcome.RESOLVED)
        self.assertTrue(outcome)
        self.assertFalse(projectile.currently_alive)

    def test_legacy_arguments_are_coerced_into_context(self):
        effects = []
        environment = CollisionEnvironment(ships=(object(),))

        context = collision_context(effects, environment)

        self.assertIs(context.effects, effects)
        self.assertIs(context.environment, environment)

    def test_consumption_properties_are_explicit(self):
        self.assertTrue(CollisionOutcome.CONSUMED_FIRST.first_consumed)
        self.assertFalse(CollisionOutcome.CONSUMED_FIRST.second_consumed)
        self.assertTrue(CollisionOutcome.CONSUMED_BOTH.first_consumed)
        self.assertTrue(CollisionOutcome.CONSUMED_BOTH.second_consumed)
        self.assertIs(
            CollisionOutcome.CONSUMED_FIRST.reversed(),
            CollisionOutcome.CONSUMED_SECOND,
        )
        self.assertIs(
            CollisionOutcome.CONSUMED_BOTH.reversed(),
            CollisionOutcome.CONSUMED_BOTH,
        )
