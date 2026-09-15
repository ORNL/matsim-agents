from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "deployments/frontier/setup/install-rocm72.sh"
VULNERABLE_BLOCK = """\
rm -rf /tmp/amd_smi_build
cp -r /opt/rocm-7.2.0/share/amd_smi /tmp/amd_smi_build
pip install /tmp/amd_smi_build --no-deps
rm -rf /tmp/amd_smi_build
"""


def installer_block() -> str:
    text = INSTALLER.read_text()
    start = text.index('echo "Installing amdsmi from ROCm 7.2..."')
    end = text.index("\n# Pre-clone triton", start)
    return text[start:end]


class RocmInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="rocm installer proof ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "trusted" / "amd_smi"
        self.prepared = self.root / "prepared"
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.fixed_name = self.shared / "amd_smi_build"
        self.trusted_marker = self.root / "trusted-executed"
        self.attacker_marker = self.root / "attacker-executed"
        self.record = self.root / "installed-path"
        self.mode = self.root / "parent-mode"
        self.target = self.root / "installed"
        self.make_package(self.source, self.trusted_marker, "trusted")
        self.make_package(self.prepared, self.attacker_marker, "attacker")
        self.prepared.chmod(0o777)

    def make_package(self, path: Path, marker: Path, label: str) -> None:
        path.mkdir(parents=True)
        (path / "pyproject.toml").write_text(
            '[build-system]\nrequires = []\nbuild-backend = "proof_backend"\nbackend-path = ["."]\n'
        )
        (path / "proof_backend.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "from zipfile import ZipFile\n"
            "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
            f"    Path({str(marker)!r}).write_text(str(os.getuid()))\n"
            f"    name = 'rocm_proof_{label}'\n"
            "    filename = name + '-0.0.0-py3-none-any.whl'\n"
            "    info = name + '-0.0.0.dist-info/'\n"
            "    with ZipFile(Path(wheel_directory) / filename, 'w') as wheel:\n"
            "        wheel.writestr(info + 'METADATA', "
            "'Metadata-Version: 2.1\\nName: ' + name + '\\nVersion: 0.0.0\\n')\n"
            "        wheel.writestr(info + 'WHEEL', "
            "'Wheel-Version: 1.0\\nGenerator: proof\\nRoot-Is-Purelib: true\\n"
            "Tag: py3-none-any\\n')\n"
            "        wheel.writestr(info + 'RECORD', '')\n"
            "    return filename\n"
        )

    def run_block(
        self, block: str, *, attack: str = "none", failure: str = "none"
    ) -> subprocess.CompletedProcess[str]:
        if attack == "symlink":
            self.fixed_name.symlink_to(self.prepared, target_is_directory=True)
        block = block.replace(
            "/opt/rocm-7.2.0/share/amd_smi", shlex.quote(str(self.source))
        ).replace("/tmp/amd_smi_build", shlex.quote(str(self.fixed_name)))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(
            PYTHON=sys.executable,
            TMPDIR=str(self.shared),
            PREPARED=str(self.prepared),
            FIXED_NAME=str(self.fixed_name),
            RECORD=str(self.record),
            MODE=str(self.mode),
            TARGET=str(self.target),
            ATTACK=attack,
            FAILURE=failure,
            PIP_CONFIG_FILE=os.devnull,
            PIP_NO_INDEX="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
        )
        script = (
            """\
set -euo pipefail
umask 000
mktemp() {
    if [ "$FAILURE" = mktemp ]; then return 25; fi
    command mktemp "$@"
}
cp() {
    if [ "$FAILURE" = cp ]; then return 24; fi
    if [ "$ATTACK" = race ]; then
        command mv -- "$PREPARED" "$FIXED_NAME"
    fi
    command cp "$@"
}
pip() {
    test "$1" = install
    shift
    printf '%s\\n' "$1" > "$RECORD"
    stat -c %a "$(dirname "$1")" > "$MODE"
    if [ "$FAILURE" = pip ]; then return 23; fi
    "$PYTHON" -m pip install --no-index --no-build-isolation --no-cache-dir \
        --disable-pip-version-check --no-compile --target "$TARGET" "$@"
}
"""
            + block
        )
        return subprocess.run(
            ["bash", "-c", script],
            env=env,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )

    def assert_success(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_trusted_install(self) -> None:
        self.assertFalse(self.attacker_marker.exists(), "Substituted package build executed")
        self.assertTrue(self.trusted_marker.exists(), "Trusted package build did not execute")
        self.assertEqual(self.trusted_marker.read_text(), str(os.getuid()))
        self.assertTrue((self.target / "rocm_proof_trusted-0.0.0.dist-info").is_dir())

    def assert_private_cleanup(self) -> None:
        installed = Path(self.record.read_text().strip())
        self.assertEqual(installed.name, "amd_smi")
        self.assertEqual(self.mode.read_text().strip(), "700")
        self.assertFalse(installed.parent.exists())

    def test_vulnerable_block_executes_attacker_build(self) -> None:
        result = self.run_block(VULNERABLE_BLOCK, attack="race")
        self.assert_success(result)
        self.assertEqual(self.attacker_marker.read_text(), str(os.getuid()))
        self.assertFalse(self.trusted_marker.exists())
        self.assertTrue((self.target / "rocm_proof_attacker-0.0.0.dist-info").is_dir())
        print("PROOF vulnerable: pip executed the substituted package build", flush=True)

    def test_vulnerable_block_legitimate_install(self) -> None:
        self.assert_success(self.run_block(VULNERABLE_BLOCK))
        self.assert_trusted_install()

    def test_fixed_block_resists_directory_substitution(self) -> None:
        self.assert_success(self.run_block(installer_block(), attack="race"))
        self.assert_trusted_install()
        self.assert_private_cleanup()
        self.assertTrue((self.fixed_name / "proof_backend.py").is_file())
        print("PROOF fixed: pip executed only the trusted package build", flush=True)

    def test_fixed_block_ignores_preexisting_symlink(self) -> None:
        self.assert_success(self.run_block(installer_block(), attack="symlink"))
        self.assert_trusted_install()
        self.assert_private_cleanup()
        self.assertTrue(self.fixed_name.is_symlink())
        self.assertTrue((self.prepared / "proof_backend.py").is_file())

    def test_fixed_block_legitimate_install(self) -> None:
        self.assert_success(self.run_block(installer_block()))
        self.assert_trusted_install()
        self.assert_private_cleanup()
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_fixed_block_cleans_up_after_pip_failure(self) -> None:
        result = self.run_block(installer_block(), failure="pip")
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assert_private_cleanup()
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_fixed_block_cleans_up_after_copy_failure(self) -> None:
        result = self.run_block(installer_block(), failure="cp")
        self.assertEqual(result.returncode, 24, result.stdout + result.stderr)
        self.assertFalse(self.record.exists())
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_fixed_block_stops_after_mktemp_failure(self) -> None:
        result = self.run_block(installer_block(), failure="mktemp")
        self.assertEqual(result.returncode, 25, result.stdout + result.stderr)
        self.assertFalse(self.record.exists())
        self.assertEqual(list(self.shared.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
