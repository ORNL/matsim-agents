from __future__ import annotations

import gzip
import io
import json
import tarfile
import zipfile

import pytest

from matsim_agents.backends.external.spectra_pka import data_fetch as df

pytest.importorskip("mendeleev")


def _tar_gz(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"pka/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "pkas.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("other-pka.tar.bz2", _tar_gz({"W184s.asc": b"OTHER LIBRARY\n"}))
        z.writestr(
            "TENDL2023n-pka.tar.bz2",
            _tar_gz(
                {
                    "Cu063s.asc": b"(n,x) proton matrix 5\n1 2\n",
                    "Cu065s.asc": b"(n,x) proton matrix 5\n3 4\n",
                    "W184s.asc": b"(n,x) proton matrix 5\n5 6\n",
                }
            ),
        )
        z.writestr("TENDL2019n-pka.tar.bz2", _tar_gz({"W184s.asc": b"2019\n"}))
    return path


def test_wanted_files_from_elements_and_nuclides():
    wanted = df.wanted_files(["Cu"], ["W184"])
    assert set(wanted) == {"Cu063s.asc", "Cu065s.asc", "W184s.asc"}


def test_extract_only_requested_files_from_matching_library(archive, tmp_path):
    dest = tmp_path / "cache"
    wanted = df.wanted_files(["Cu"], ["W184", "Fe056"][:1])
    with archive.open("rb") as stream:
        report = df.extract(stream, dest, wanted, "TENDL2023n", source="file:pkas.zip")
    assert report["missing"] == []
    files = sorted(p.name for p in (dest / "TENDL2023n-pka").glob("*.asc"))
    assert files == ["Cu063s.asc", "Cu065s.asc", "W184s.asc"]
    assert (dest / "TENDL2023n-pka" / "W184s.asc").read_bytes().endswith(b"5 6\n")
    assert not (dest / "other-pka").exists()
    manifest = json.loads((dest / "TENDL2023n-pka" / "manifest.json").read_text())
    entry = manifest["files"]["W184s.asc"]
    assert entry["target_isotope"] == "W184" and entry["checksum_verified"] is False
    assert len(entry["sha256"]) == 64


def test_missing_isotope_reported_and_cache_reused(archive, tmp_path):
    dest = tmp_path / "cache"
    wanted = df.wanted_files([], ["W184", "W999"])
    with archive.open("rb") as stream:
        report = df.extract(stream, dest, wanted, "TENDL2023n", source="s")
    assert report["missing"] == ["W999s.asc"]
    with archive.open("rb") as stream:
        again = df.extract(stream, dest, {"W184s.asc": ("W", 184)}, "TENDL2023n", source="s")
    assert again["found"]["TENDL2023n-pka/W184s.asc"]["status"] == "cached"


def test_main_dry_run_and_missing_exit_code(archive, tmp_path, capsys):
    assert df.main(["--dest", str(tmp_path), "--elements", "Cu", "--dry-run"]) == 0
    assert "Cu063s.asc" in capsys.readouterr().out
    code = df.main(["--dest", str(tmp_path / "c"), "--nuclides", "W999", "--archive", str(archive)])
    assert code == 2


def _zip_of(tmp_path, members):
    path = tmp_path / "m.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_unknown_library_is_an_error(archive, tmp_path):
    with archive.open("rb") as stream, pytest.raises(df.FetchError):
        df.extract(stream, tmp_path / "c", {"W184s.asc": ("W", 184)}, "TENDL", source="s")


def test_traversal_member_name_stays_inside_destination(tmp_path):
    zpath = _zip_of(tmp_path, {"../../evil.tar.bz2": _tar_gz({"W184s.asc": b"x\n"})})
    dest = tmp_path / "cache"
    with zpath.open("rb") as stream, pytest.raises(df.FetchError):
        df.extract(stream, dest, {"W184s.asc": ("W", 184)}, "..", source="s")
    assert not (tmp_path / "evil").exists()


def test_modified_cached_file_is_re_extracted(archive, tmp_path):
    dest = tmp_path / "cache"
    wanted = {"W184s.asc": ("W", 184)}
    with archive.open("rb") as stream:
        df.extract(stream, dest, wanted, "TENDL2023n", source="s")
    target = dest / "TENDL2023n-pka" / "W184s.asc"
    target.write_bytes(b"corrupt")
    with archive.open("rb") as stream:
        again = df.extract(stream, dest, wanted, "TENDL2023n", source="s")
    assert again["found"]["TENDL2023n-pka/W184s.asc"]["status"] == "downloaded"
    assert target.read_bytes().endswith(b"5 6\n")


def test_missing_manifest_entry_is_re_extracted(archive, tmp_path):
    dest = tmp_path / "cache"
    wanted = {"W184s.asc": ("W", 184)}
    lib = dest / "TENDL2023n-pka"
    lib.mkdir(parents=True)
    (lib / "W184s.asc").write_bytes(b"partial")
    with archive.open("rb") as stream:
        report = df.extract(stream, dest, wanted, "TENDL2023n", source="s")
    assert report["found"]["TENDL2023n-pka/W184s.asc"]["status"] == "downloaded"
    assert "W184s.asc" in json.loads((lib / "manifest.json").read_text())["files"]


def test_bzip2_member_of_another_library_is_skipped(tmp_path):
    import bz2

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo("pka/W184s.asc")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"b\n"))
    zpath = _zip_of(
        tmp_path,
        {
            "TENDL2017n-pka.tar.bz2": bz2.compress(raw.getvalue()),
            "TENDL2023n-pka.tar.bz2": _tar_gz({"W184s.asc": b"g\n"}),
        },
    )
    with zpath.open("rb") as stream:
        report = df.extract(
            stream, tmp_path / "c", {"W184s.asc": ("W", 184)}, "TENDL2023n", source="s"
        )
    assert report["missing"] == []
