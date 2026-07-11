import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _literal_keyword_from_call(tree, call_name, keyword_name):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Name) or function.id != call_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"{call_name}(..., {keyword_name}=...) was not found")


def _spec_tree(filename):
    return ast.parse((PROJECT_ROOT / filename).read_text(encoding="utf-8"))


class CpuAIBuildConfigurationTests(unittest.TestCase):
    def test_lightweight_spec_still_excludes_torch_packages(self):
        excludes = _literal_keyword_from_call(
            _spec_tree("StarAI.spec"),
            "Analysis",
            "excludes",
        )

        self.assertIn("torch", excludes)
        self.assertIn("torchvision", excludes)
        self.assertIn("torchaudio", excludes)

    def test_cpuai_spec_bundles_torch_but_excludes_companion_packages(self):
        excludes = _literal_keyword_from_call(
            _spec_tree("StarAI_CPUAI.spec"),
            "Analysis",
            "excludes",
        )

        self.assertNotIn("torch", excludes)
        self.assertIn("torchvision", excludes)
        self.assertIn("torchaudio", excludes)

    def test_cpuai_spec_uses_cpuai_executable_and_collect_names(self):
        tree = _spec_tree("StarAI_CPUAI.spec")

        self.assertEqual(
            _literal_keyword_from_call(tree, "EXE", "name"),
            "StarAI_CPUAI",
        )
        self.assertEqual(
            _literal_keyword_from_call(tree, "COLLECT", "name"),
            "StarAI_CPUAI",
        )

    def test_cpuai_command_targets_cpuai_build_name(self):
        command = (PROJECT_ROOT / "buildcpu.cmd").read_text(encoding="utf-8")

        self.assertIn('-BuildName "StarAI_CPUAI"', command)
        self.assertIn(".venv-cpuai\\Scripts\\python.exe", command)
        self.assertIn("-PythonPath", command)
        self.assertIn("build.ps1", command)

    def test_gpu_command_targets_train_build_with_main_venv(self):
        command = (PROJECT_ROOT / "buildgpu.cmd").read_text(encoding="utf-8")

        self.assertIn('-BuildName "StarAI_Train"', command)
        self.assertIn(".venv\\Scripts\\python.exe", command)
        self.assertIn("-PythonPath", command)
        self.assertIn("build.ps1", command)

    def test_only_canonical_build_commands_are_exposed(self):
        commands = {path.name for path in PROJECT_ROOT.glob("build*.cmd")}

        self.assertEqual(commands, {"build.cmd", "buildcpu.cmd", "buildgpu.cmd"})

    def test_build_script_keeps_default_venv_and_allows_python_path(self):
        script = (PROJECT_ROOT / "build.ps1").read_text(encoding="utf-8")

        self.assertIn('[string]$PythonPath', script)
        self.assertIn('".venv\\Scripts\\python.exe"', script)
        self.assertIn("if ($PythonPath)", script)
        self.assertIn("Resolve-Path -LiteralPath $PythonPath", script)
        self.assertIn("requirements-build-tools.txt", script)

    def test_build_tools_requirements_do_not_pull_runtime_torch(self):
        requirements = (
            PROJECT_ROOT / "requirements-build-tools.txt"
        ).read_text(encoding="utf-8").splitlines()

        self.assertIn("pyinstaller==6.21.0", requirements)
        self.assertFalse(any(line.startswith("-r ") for line in requirements))
        self.assertFalse(any("torch" in line for line in requirements))

    def test_cpuai_requirements_use_cpu_torch_wheel_index(self):
        requirements = (
            PROJECT_ROOT / "requirements-cpuai.txt"
        ).read_text(encoding="utf-8").splitlines()

        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu", requirements)
        self.assertIn("pygame-ce==2.5.2", requirements)
        self.assertIn("numpy==2.5.1", requirements)
        self.assertIn("torch==2.7.0+cpu", requirements)
        self.assertFalse(any("cu128" in line for line in requirements))
        self.assertFalse(any(line.startswith("torchvision") for line in requirements))
        self.assertFalse(any(line.startswith("torchaudio") for line in requirements))


if __name__ == "__main__":
    unittest.main()
