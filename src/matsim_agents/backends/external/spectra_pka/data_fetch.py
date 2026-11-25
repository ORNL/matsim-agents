"""Extract only the needed PKA recoil matrices from the official SPECTRA-PKA ``pkas.zip``.

The archive (about 7.9 GB, see docs/external/spectra-pka-data.md) is a ZIP of stored gzip
tarballs, one per library (for example ``TENDL2023n-pka.tar.bz2``, which is gzip despite the
name), each holding ``pka/<El><AAA>s.asc`` files of 20-90 MB. The server does not support range
requests, so the archive is streamed once and only the requested files are written to disk.
Material-independent: the isotope list comes from elements (natural abundance), an explicit
composition, or explicit nuclides.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import struct
import sys
import tarfile
import urllib.request
import zlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from matsim_agents.backends.external.spectra_pka.isotopes import (
    IsotopeError,
    natural_abundance,
    parse_nuclide,
)
from matsim_agents.execution.locking import exclusive_lock

OFFICIAL_URL = (
    "https://www.dropbox.com/scl/fo/jqgdshufqxvwea8xzecwf/"
    "ACablw3sL4Ru0rgiogr17Xs?rlkey=ryqsvfgjm4u6wzoxaqcmkteip&dl=1"
)
MANIFEST = "manifest.json"
CHUNK = 1 << 20
_LOCAL, _CENTRAL = b"PK\x03\x04", b"PK\x01\x02"


class FetchError(RuntimeError):
    """The archive cannot be read or does not contain what was asked for."""


def matrix_name(element: str, mass: int) -> str:
    return f"{element}{mass:03d}s.asc"


def wanted_files(
    elements: list[str], nuclides: list[str], composition: dict[str, float] | None = None
) -> dict[str, tuple[str, int]]:
    """File name -> (element, mass number) for every isotope the material can contain."""
    isotopes: set[tuple[str, int]] = set()
    for element in [*elements, *(composition or {})]:
        isotopes.update((element, a) for a in natural_abundance(element))
    for label in nuclides:
        isotopes.add(parse_nuclide(label))
    if not isotopes:
        raise IsotopeError("no elements, composition or nuclides given")
    return {matrix_name(e, a): (e, a) for e, a in sorted(isotopes)}


class _Stream:
    def __init__(self, raw: IO[bytes]) -> None:
        self.raw, self.buf = raw, b""

    def read(self, n: int) -> bytes:
        if self.buf:
            out, self.buf = self.buf[:n], self.buf[n:]
            return out
        return self.raw.read(n)

    def read_exact(self, n: int) -> bytes:
        parts: list[bytes] = []
        while n > 0:
            chunk = self.read(n)
            if not chunk:
                break
            parts.append(chunk)
            n -= len(chunk)
        return b"".join(parts)

    def unread(self, data: bytes) -> None:
        self.buf = data + self.buf

    def resync(self, data: bytes) -> None:
        """Skip the zip data descriptor: advance to the next record signature."""
        while True:
            hits = [i for i in (data.find(_LOCAL), data.find(_CENTRAL)) if i >= 0]
            if hits:
                self.unread(data[min(hits) :])
                return
            more = self.read(CHUNK)
            if not more:
                return
            data = data[-3:] + more


class _Gunzip:
    """File-like view of one stored gzip member; knows where the member ends."""

    def __init__(self, src: _Stream) -> None:
        magic = src.read_exact(3)
        src.unread(magic)
        self.z: Any = bz2.BZ2Decompressor() if magic == b"BZh" else zlib.decompressobj(47)
        self.src, self.pending = src, b""

    def read(self, n: int) -> bytes:
        while len(self.pending) < n and not self.z.eof:
            chunk = self.src.read(CHUNK)
            if not chunk:
                raise FetchError("archive ended inside a member (truncated download?)")
            self.pending += self.z.decompress(chunk)
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def finish(self) -> None:
        while not self.z.eof:
            self.read(CHUNK)
            self.pending = b""
        self.src.resync(self.z.unused_data)


def _members(src: _Stream) -> Iterator[tuple[str, int]]:
    while src.read_exact(4) == _LOCAL:
        header = src.read_exact(26)
        method = struct.unpack("<H", header[4:6])[0]
        name_len, extra_len = struct.unpack("<HH", header[22:26])
        name = src.read_exact(name_len).decode("utf-8", "replace")
        src.read_exact(extra_len)
        yield name, method


def library_id(member: str) -> str:
    return member.split(".tar")[0]


def _extract_one(tar: tarfile.TarFile, info: tarfile.TarInfo, target: Path) -> dict[str, Any]:
    handle = tar.extractfile(info)
    assert handle is not None
    digest, size, first = hashlib.sha256(), 0, b""
    part = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        with part.open("wb") as out:
            while chunk := handle.read(CHUNK):
                first = first or chunk[:200]
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "header": first.decode("ascii", "replace").splitlines()[0].strip() if first else "",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _is_cached(lib_dir: Path, base: str, target: Path) -> bool:
    manifest = lib_dir / MANIFEST
    if not (target.is_file() and manifest.is_file()):
        return False
    try:
        entry = json.loads(manifest.read_text())["files"][base]
    except (ValueError, KeyError):
        return False
    return entry.get("bytes") == target.stat().st_size and entry.get("sha256") == _sha256(target)


def _record(lib_dir: Path, library: str, base: str, entry: dict[str, Any]) -> None:
    manifest_path = lib_dir / MANIFEST
    doc = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    doc.setdefault("library", library)
    doc.setdefault("files", {})[base] = entry
    tmp = manifest_path.with_name(f".{MANIFEST}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path)


def extract(
    stream: IO[bytes],
    destination: Path,
    wanted: dict[str, tuple[str, int]],
    library: str,
    *,
    source: str,
    force: bool = False,
) -> dict[str, Any]:
    """Stream the archive and write the wanted files of the one library named ``library``."""
    src = _Stream(stream)
    todo = set(wanted)
    found: dict[str, dict[str, Any]] = {}
    seen: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    matched = False
    for name, method in _members(src):
        if name.endswith("/"):
            continue
        seen.append(name)
        if method != 0:
            raise FetchError(f"{name}: compressed zip members are not supported")
        gz = _Gunzip(src)
        lid = library_id(name)
        if library.lower() not in {lid.lower(), lid.removesuffix("-pka").lower()}:
            gz.finish()
            continue
        matched = True
        safe = os.path.basename(lid.replace("\\", "/"))
        if safe in {"", ".", ".."}:
            raise FetchError(f"unsafe archive member name {name!r}")
        lib_dir = destination / safe
        lib_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=gz, mode="r|") as tar:  # type: ignore[call-overload]
            for info in tar:
                base = os.path.basename(info.name)
                if not info.isfile() or base not in todo:
                    continue
                target = lib_dir / base
                with exclusive_lock(lib_dir / ".lock", timeout=300):
                    if not force and _is_cached(lib_dir, base, target):
                        found[f"{safe}/{base}"] = {"status": "cached", "bytes": 0}
                    else:
                        meta = _extract_one(tar, info, target)
                        meta["status"] = "downloaded"
                        meta["retrieved_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
                        element, mass = wanted[base]
                        entry = {
                            **meta,
                            "file": base,
                            "target_isotope": f"{element}{mass}",
                            "source": source,
                            "checksum_verified": False,
                            "note": "SHA-256 computed locally; the publisher provides no checksums",
                        }
                        _record(lib_dir, safe, base, entry)
                        found[f"{safe}/{base}"] = meta
                todo.discard(base)
                if not todo:
                    return _finish(wanted, found, seen)
        gz.finish()
    if not matched:
        raise FetchError(f"no archive member is the library {library!r}; seen: {seen}")
    return _finish(wanted, found, seen)


def _finish(
    wanted: dict[str, tuple[str, int]], found: dict[str, dict[str, Any]], seen: list[str]
) -> dict[str, Any]:
    got = {key.split("/", 1)[1] for key in found}
    return {"found": found, "missing": sorted(set(wanted) - got), "archive_members_seen": seen}


def open_source(url: str | None, archive: Path | None) -> tuple[IO[bytes], str]:
    if archive is not None:
        return archive.open("rb"), f"file:{archive.name}"
    target = url or OFFICIAL_URL
    request = urllib.request.Request(target, headers={"User-Agent": "matsim-agents"})
    return urllib.request.urlopen(request, timeout=120), target.split("?")[0]  # noqa: S310


def _parse_composition(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        element, _, fraction = part.partition(":")
        out[element.strip()] = float(fraction) if fraction else 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract needed SPECTRA-PKA recoil matrices from the official pkas.zip"
    )
    p.add_argument("--dest", type=Path, required=True, help="cache directory (shared is fine)")
    p.add_argument("--library", default="TENDL2023n", help="library name, e.g. TENDL2023n (exact)")
    p.add_argument("--elements", default="", help="comma list; all natural isotopes of each")
    p.add_argument("--composition", default="", help="El:fraction,... (natural isotopes of each)")
    p.add_argument("--nuclides", default="", help="comma list such as W184,Fe56 (added)")
    p.add_argument("--archive", type=Path, help="read a local pkas.zip instead of downloading")
    p.add_argument("--url", help="override the official download URL")
    p.add_argument("--force", action="store_true", help="re-extract files already present")
    p.add_argument(
        "--allow-missing", action="store_true", help="exit 0 even if isotopes are absent"
    )
    p.add_argument("--dry-run", action="store_true", help="print the required files and exit")
    args = p.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    try:
        wanted = wanted_files(
            split(args.elements),
            split(args.nuclides),
            _parse_composition(args.composition) if args.composition else None,
        )
    except (IsotopeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(wanted)} matrices needed: {' '.join(wanted)}")
    if args.dry_run:
        return 0
    stream, source = open_source(args.url, args.archive)
    try:
        report = extract(stream, args.dest, wanted, args.library, source=source, force=args.force)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        stream.close()
    for key, meta in sorted(report["found"].items()):
        print(f"{meta['status']:10s} {key}")
    if report["missing"]:
        print("MISSING from this library: " + " ".join(report["missing"]), file=sys.stderr)
        return 0 if args.allow_missing else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
