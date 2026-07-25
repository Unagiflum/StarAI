from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from src.training import windows_qos
from src.training.coordinated import CoordinatedTrainingSession
from src.training.process_session import independent_training_process_main
from src.training.process_worker import worker_process_main
from src.training.session import TrainingSession


class WindowsTrainingQosTests(unittest.TestCase):
    def tearDown(self):
        windows_qos._process_qos_users = 0
        windows_qos._thread_qos_state.users = 0

    @mock.patch("src.training.windows_qos.ctypes.WinDLL", create=True)
    @mock.patch("src.training.windows_qos.sys.platform", "win32")
    def test_process_high_qos_disables_execution_speed_throttling(self, win_dll):
        kernel32 = win_dll.return_value
        kernel32.GetCurrentProcess.return_value = 123
        kernel32.SetProcessInformation.return_value = 1

        changed = windows_qos.request_current_process_high_qos()

        self.assertTrue(changed)
        args = kernel32.SetProcessInformation.call_args.args
        self.assertEqual(args[0], 123)
        self.assertEqual(args[1], windows_qos._PROCESS_POWER_THROTTLING)
        state = args[2]._obj
        self.assertEqual(state.Version, 1)
        self.assertEqual(
            state.ControlMask,
            windows_qos._POWER_THROTTLING_EXECUTION_SPEED,
        )
        self.assertEqual(state.StateMask, 0)
        self.assertEqual(args[3], 12)

    @mock.patch("src.training.windows_qos.ctypes.WinDLL", create=True)
    @mock.patch("src.training.windows_qos.sys.platform", "win32")
    def test_thread_high_qos_disables_execution_speed_throttling(self, win_dll):
        kernel32 = win_dll.return_value
        kernel32.GetCurrentThread.return_value = 456
        kernel32.SetThreadInformation.return_value = 1

        changed = windows_qos.request_current_thread_high_qos()

        self.assertTrue(changed)
        args = kernel32.SetThreadInformation.call_args.args
        self.assertEqual(args[0], 456)
        self.assertEqual(args[1], windows_qos._THREAD_POWER_THROTTLING)
        state = args[2]._obj
        self.assertEqual(state.Version, 1)
        self.assertEqual(
            state.ControlMask,
            windows_qos._POWER_THROTTLING_EXECUTION_SPEED,
        )
        self.assertEqual(state.StateMask, 0)
        self.assertEqual(args[3], 12)

    @mock.patch("src.training.windows_qos.sys.platform", "linux")
    def test_non_windows_requests_are_noops(self):
        self.assertFalse(windows_qos.request_current_process_high_qos())
        self.assertFalse(windows_qos.request_current_thread_high_qos())

    def test_nested_training_runs_share_and_restore_process_policy(self):
        with (
            mock.patch.object(
                windows_qos,
                "request_current_process_high_qos",
                return_value=True,
            ) as request_process,
            mock.patch.object(
                windows_qos,
                "request_current_thread_high_qos",
                return_value=True,
            ) as request_thread,
            mock.patch.object(
                windows_qos,
                "_restore_current_process_qos",
                return_value=True,
            ) as restore_process,
            mock.patch.object(
                windows_qos,
                "_restore_current_thread_qos",
                return_value=True,
            ) as restore_thread,
        ):
            with windows_qos.training_high_qos():
                with windows_qos.training_high_qos():
                    self.assertEqual(windows_qos._process_qos_users, 2)
                self.assertEqual(windows_qos._process_qos_users, 1)

        request_process.assert_called_once_with()
        request_thread.assert_called_once_with()
        restore_thread.assert_called_once_with()
        restore_process.assert_called_once_with()
        self.assertEqual(windows_qos._process_qos_users, 0)

    def test_failed_process_request_does_not_suppress_training_error(self):
        with (
            mock.patch.object(
                windows_qos,
                "request_current_process_high_qos",
                return_value=False,
            ),
            mock.patch.object(
                windows_qos,
                "request_current_thread_high_qos",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with windows_qos.training_high_qos():
                    raise RuntimeError("training failed")


class TrainingQosIntegrationTests(unittest.TestCase):
    def test_gpu_training_thread_uses_high_qos_context(self):
        session = SimpleNamespace(
            _run_loop=mock.Mock(),
            _mark_stopped=mock.Mock(),
        )
        context = mock.MagicMock()

        with mock.patch(
            "src.training.session.training_high_qos",
            return_value=context,
        ) as high_qos:
            TrainingSession._run_worker(session)

        high_qos.assert_called_once_with()
        context.__enter__.assert_called_once_with()
        context.__exit__.assert_called_once()
        session._run_loop.assert_called_once_with()
        session._mark_stopped.assert_called_once_with()

    def test_coordinated_training_keeps_low_priority_and_uses_high_qos(self):
        session = SimpleNamespace(_run_worker_high_qos=mock.Mock())
        context = mock.MagicMock()

        with (
            mock.patch(
                "src.training.coordinated._set_current_thread_below_normal_priority"
            ) as low_priority,
            mock.patch(
                "src.training.coordinated.training_high_qos",
                return_value=context,
            ) as high_qos,
        ):
            CoordinatedTrainingSession._run_worker(session)

        low_priority.assert_called_once_with()
        high_qos.assert_called_once_with()
        session._run_worker_high_qos.assert_called_once_with()

    def test_coordinated_cpu_worker_process_requests_high_qos(self):
        connection = mock.Mock()
        connection.recv.side_effect = EOFError
        worker = mock.Mock()

        with (
            mock.patch(
                "src.training.process_worker._set_worker_process_below_normal_priority"
            ) as low_priority,
            mock.patch(
                "src.training.process_worker.request_current_process_high_qos"
            ) as process_qos,
            mock.patch(
                "src.training.process_worker.request_current_thread_high_qos"
            ) as thread_qos,
            mock.patch(
                "src.training.process_worker.CoordinatedSimulationWorker",
                return_value=worker,
            ),
        ):
            worker_process_main(connection)

        low_priority.assert_called_once_with()
        process_qos.assert_called_once_with()
        thread_qos.assert_called_once_with()
        worker.close.assert_called_once_with()

    def test_independent_cpu_process_requests_high_qos_before_training_setup(self):
        with (
            mock.patch(
                "src.training.process_session._set_worker_process_below_normal_priority"
            ) as low_priority,
            mock.patch(
                "src.training.process_session.request_current_process_high_qos"
            ) as process_qos,
            mock.patch(
                "src.training.process_session.request_current_thread_high_qos"
            ) as thread_qos,
            mock.patch(
                "src.training.process_session.torch_backend.get_torch",
                side_effect=RuntimeError("stop after policy setup"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "policy setup"):
                independent_training_process_main(
                    bundled_dir=Path("bundled"),
                    user_dir=Path("user"),
                    slot=None,
                    metadata={},
                    config=None,
                    batch_grouping=1,
                    stop_at_batch=None,
                    stop_at_epsilon=None,
                    initial_history=(),
                    initial_log_lines=(),
                    save_coordinator=None,
                    message_queue=None,
                    control_queue=None,
                    stop_event=None,
                    display_event=None,
                    display_frame_count=1,
                )

        low_priority.assert_called_once_with()
        process_qos.assert_called_once_with()
        thread_qos.assert_called_once_with()

    def test_independent_cpu_process_starts_normal_when_display_is_enabled(self):
        display_event = mock.Mock()
        display_event.is_set.return_value = True

        with (
            mock.patch(
                "src.training.process_session._set_worker_process_below_normal_priority"
            ) as low_priority,
            mock.patch(
                "src.training.process_session._set_worker_process_normal_priority"
            ) as normal_priority,
            mock.patch(
                "src.training.process_session.request_current_process_high_qos"
            ),
            mock.patch(
                "src.training.process_session.request_current_thread_high_qos"
            ),
            mock.patch(
                "src.training.process_session.torch_backend.get_torch",
                side_effect=RuntimeError("stop after priority setup"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "priority setup"):
                independent_training_process_main(
                    bundled_dir=Path("bundled"),
                    user_dir=Path("user"),
                    slot=None,
                    metadata={},
                    config=None,
                    batch_grouping=1,
                    stop_at_batch=None,
                    stop_at_epsilon=None,
                    initial_history=(),
                    initial_log_lines=(),
                    save_coordinator=None,
                    message_queue=None,
                    control_queue=None,
                    stop_event=None,
                    display_event=display_event,
                    display_frame_count=1,
                )

        low_priority.assert_not_called()
        normal_priority.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
