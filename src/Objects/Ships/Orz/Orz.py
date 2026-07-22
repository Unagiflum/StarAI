import src.const as const
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Orz.A1.OrzA1 import OrzA1
from src.Objects.Ships.Orz.A2.OrzA2 import OrzA2
from src.Objects.Ships.Orz.A3.OrzA3 import OrzA3
from src.Objects.Ships.space_ship import SpaceShip
from src.resources import centered_overlay


class Orz(SpaceShip):
    action_factories = {1: OrzA1}

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.turret = OrzA2(self)
        self._turret_composites = {}
        self.active_marines = []
        self._marine_trajectory_state = None
        self._marine_trajectory_horizon = 0
        self._marine_target_trajectory = ()

    @property
    def turret_heading(self):
        return self.turret.absolute_heading

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.turret.reset()
        self._clear_marine_trajectory_cache()

    def predict_marine_target_trajectory(self, target, frames):
        """Share one immutable target trajectory among this ship's marines.

        Preserved abilities can update on either side of ships in the world's
        stable order.  Keying by the target's prediction-relevant state, rather
        than only the battle frame, prevents reuse across a mid-frame target
        update while still allowing every marine seeing the same state to share
        the result.
        """
        frames = int(frames)
        if frames <= 0:
            return ()

        state = self._marine_target_prediction_state(target)
        if (
            state == self._marine_trajectory_state
            and self._marine_trajectory_horizon >= frames
        ):
            return self._marine_target_trajectory[:frames]

        trajectory = target.predict_unhindered_trajectory(frames=frames)
        immutable_trajectory = tuple(tuple(position) for position in trajectory)
        self._marine_trajectory_state = state
        self._marine_trajectory_horizon = frames
        self._marine_target_trajectory = immutable_trajectory
        return immutable_trajectory

    @staticmethod
    def _marine_target_prediction_state(target):
        planet = getattr(target, "planet", None)
        bound_predictor = getattr(target, "predict_unhindered_trajectory", None)
        predictor = getattr(bound_predictor, "__func__", bound_predictor)

        def vector(name):
            value = getattr(target, name, None)
            return None if value is None else tuple(value)

        return (
            id(target),
            predictor,
            vector("position"),
            vector("velocity"),
            vector("accumulated_impulses"),
            vector("collision_velocity"),
            getattr(target, "inertia", None),
            getattr(target, "heading", None),
            getattr(target, "max_thrust", None),
            getattr(target, "can_expire", None),
            getattr(target, "expiration_timer", None),
            id(planet) if planet is not None else None,
            tuple(planet.position) if planet is not None else None,
            getattr(planet, "gravity", None),
            getattr(planet, "diameter", None),
            const.ARENA_SIZE,
            const.SPEED_SCALE,
            const.SPEED_LIMIT,
            const.GRAVITY_RANGE,
            const.GRAVITY_MULTIPLIER,
            const.TURN_ANGLE,
        )

    def _clear_marine_trajectory_cache(self):
        self._marine_trajectory_state = None
        self._marine_trajectory_horizon = 0
        self._marine_target_trajectory = ()

    def plan_action1(self):
        if self.action2_active:
            return ActionPlan.invalid(1)
        return self.validate_action(1, self.action_factories[1])

    def plan_action2(self):
        turn_left = self.turn_left_active
        turn_right = self.turn_right_active
        if not self.can_action2() or turn_left == turn_right:
            return ActionPlan.invalid(2)
        direction = 1 if turn_right else -1
        return self.prepare_action_plan(
            2,
            side_effects=(lambda: self.turret.turn(direction),),
            use_first_object_sound=False,
        )

    def turn_input_enabled(self):
        # A2 changes the direction keys from hull steering to turret steering.
        return not self.action2_active

    def plan_action3(self):
        self.active_marines = [
            marine for marine in self.active_marines if marine.currently_alive
        ]
        opponent = self.opponent
        if opponent is None:
            # Preserve the existing combined-input cooldown outside a bound
            # battle (menus and characterization tests have no opponent).
            return self.validate_action(3)
        if (
            not self.can_action3()
            or self.current_hp <= 1
            or len(self.active_marines) >= OrzA3.MAX_MARINES
            or not opponent.currently_alive
            or opponent.current_hp <= 0
            or not opponent.trackable
        ):
            return ActionPlan.invalid(3)

        marine = OrzA3(self)
        return self.prepare_action_plan(
            3,
            marine,
            crew_change=-1,
            side_effects=(lambda: self.active_marines.append(marine),),
        )

    def handles_combined_action(self):
        return True

    def attach_limpet(self, *, source=None):
        super().attach_limpet(source=source)
        self._turret_composites.clear()

    def set_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index

        sprite_idx = interpolated_sprite_index(self, interp_t)
        turret_sprite = self.turret.get_sprite(interp_t)
        key = (sprite_idx, id(turret_sprite))
        if key not in self._turret_composites:
            self._turret_composites[key] = centered_overlay(
                self.sprites[sprite_idx],
                turret_sprite,
            )
        return self._turret_composites[key]
