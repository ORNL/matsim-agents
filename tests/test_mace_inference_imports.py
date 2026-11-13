from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "mace-inference.sh"
BOOTSTRAP = """
import sys
from types import ModuleType
from packaging.requirements import Requirement
from matsim_agents.backends.mlip.mace_inference import configure_mace_inference_imports

# Test the shim contract without requiring optional ML packages in CI.
class RequirementCache:
    def __init__(self, requirement):
        self.requirement = requirement
    def _check_requirement(self):
        self.available = True
        self.message = "original check"
    def __bool__(self):
        self._check_requirement()
        return self.available

imports = ModuleType("lightning_utilities.core.imports")
imports.Requirement = Requirement
imports.RequirementCache = RequirementCache
sys.modules["lightning_utilities.core.imports"] = imports
"""


def run_code(code, *, enabled=True, launcher=False):
    env = dict(os.environ)
    env.pop("MATSIM_MACE_SKIP_TRANSFORMERS", None)
    if enabled and not launcher:
        env["MATSIM_MACE_SKIP_TRANSFORMERS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    command = [sys.executable, "-c", BOOTSTRAP + "\n" + textwrap.dedent(code)]
    if launcher:
        command = ["bash", str(LAUNCHER), *command]
    return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=False)


def assert_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_does_not_change_imports():
    assert_success(
        run_code(
            """
        original = RequirementCache._check_requirement
        finders = list(sys.meta_path)
        configure_mace_inference_imports()
        assert RequirementCache._check_requirement is original
        assert sys.meta_path == finders
    """,
            enabled=False,
        )
    )


def test_launcher_disables_only_transformers_and_is_idempotent():
    result = run_code(
        """
        configure_mace_inference_imports()
        finders = list(sys.meta_path)
        configure_mace_inference_imports()
        assert sys.meta_path == finders
        for requirement in ("transformers", "transformers>=4.4.0", "transformers[text]"):
            assert not RequirementCache(requirement)
        assert RequirementCache("torch>=2")
        try:
            import transformers
        except ModuleNotFoundError as exc:
            assert "disabled" in str(exc)
        else:
            raise AssertionError("Transformers must not be importable")
        assert not any(n.startswith("transformers") for n in sys.modules)
    """,
        launcher=True,
    )
    assert_success(result)
    assert result.stderr.count("MACE inference isolation active") == 1


@pytest.mark.parametrize("module", ["mace", "torchmetrics", "transformers"])
def test_rejects_late_activation(module):
    result = run_code(f"""
        sys.modules[{module!r}] = ModuleType({module!r})
        configure_mace_inference_imports()
    """)
    assert result.returncode != 0
    assert "start a fresh process" in result.stderr


def test_rejects_invalid_opt_in():
    result = run_code("""
        import os
        os.environ["MATSIM_MACE_SKIP_TRANSFORMERS"] = "yes"
        configure_mace_inference_imports()
    """)
    assert result.returncode != 0
    assert "must be unset or '1'" in result.stderr


def test_launcher_flag_reaches_spawned_workers(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        BOOTSTRAP
        + textwrap.dedent("""
        import multiprocessing
        import os
        def worker():
            assert os.environ["MATSIM_MACE_SKIP_TRANSFORMERS"] == "1"
            configure_mace_inference_imports()
            assert not RequirementCache("transformers>=4.4")
        if __name__ == "__main__":
            child = multiprocessing.get_context("spawn").Process(target=worker)
            child.start()
            child.join(20)
            if child.is_alive():
                child.terminate()
                child.join()
                raise AssertionError("Worker timed out")
            assert child.exitcode == 0
    """)
    )
    env = dict(os.environ)
    env.pop("MATSIM_MACE_SKIP_TRANSFORMERS", None)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    assert_success(
        subprocess.run(
            ["bash", str(LAUNCHER), sys.executable, str(worker)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    )


def test_launcher_requires_command():
    result = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
