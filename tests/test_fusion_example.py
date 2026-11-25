import importlib.util
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "examples" / "fusion_pfc" / "synthetic_demo.py"


def test_synthetic_demo_runs(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("fusion_demo", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(tmp_path)
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out.upper()
    assert "nothing dispatched" in out
    assert (tmp_path / "ledger.jsonl").read_text().strip()
