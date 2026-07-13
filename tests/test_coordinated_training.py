import time
import unittest
from pathlib import Path

from src.training.coordinated import (
    CoordinatedRuntimeComponents,
    CoordinatedTrainingRecord,
    CoordinatedTrainingSession,
)
from src.training.model_registry import (
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import TrainingOrchestrationConfig
from src.training.replay import TrainingReplayBuffer


def _record(instance_id, ship):
    metadata = metadata_from_state(
        ship=ship,
        slot=1,
        description=f"{ship} test",
        architecture=model_architecture_metadata(8, 1),
        training={},
    )
    return CoordinatedTrainingRecord(
        instance_id=instance_id,
        repository=TrainingModelRepository(Path("unused"), Path("unused")),
        slot=TrainingModelSlot(ship, 1, SLOT_USER, metadata=metadata),
        metadata=metadata,
        config=TrainingOrchestrationConfig(
            trainee_ship=ship,
            hidden_layer_width=8,
            hidden_layer_count=1,
            training_device="cpu",
        ),
        batch_grouping=1,
    )


class CoordinatedTrainingSessionTests(unittest.TestCase):
    def _session(self, *, component_builder=None):
        return CoordinatedTrainingSession(
            (_record(1, "Earthling"), _record(2, "Androsynth")),
            component_builder=component_builder or self._component_builder([]),
            idle_sleep_seconds=0.001,
        )

    def _component_builder(self, built):
        def build(record):
            built.append(record.instance_id)
            return CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=TrainingReplayBuffer(4),
            )

        return build

    def test_start_builds_records_and_stop_all_marks_every_proxy_stopped(self):
        built = []
        session = self._session(component_builder=self._component_builder(built))

        session.start()
        deadline = time.time() + 1.0
        while len(built) < 2 and time.time() < deadline:
            time.sleep(0.005)

        proxies = session.proxies
        self.assertEqual(built, [1, 2])
        self.assertTrue(proxies[1].status.running)
        self.assertTrue(proxies[2].status.running)
        self.assertEqual(proxies[1].status.display_message, "Coordinated scheduler idle")

        proxies[1].request_stop()
        session.join(1.0)

        self.assertFalse(session.active)
        self.assertFalse(proxies[1].status.running)
        self.assertFalse(proxies[1].status.stopping)
        self.assertFalse(proxies[2].status.running)

    def test_component_build_error_marks_all_records_and_exits(self):
        def fail(_record):
            raise RuntimeError("component build failed")

        session = self._session(component_builder=fail)

        session.start()
        session.join(1.0)

        self.assertFalse(session.active)
        self.assertEqual(session.status_for_instance(1).error, "component build failed")
        self.assertEqual(session.status_for_instance(2).error, "component build failed")


if __name__ == "__main__":
    unittest.main()
