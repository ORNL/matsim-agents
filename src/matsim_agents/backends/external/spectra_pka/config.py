"""Configuration for the external SPECTRA-PKA interface.

Only fields that the adapter actually implements are accepted (``extra="forbid"``).
SPECTRA-PKA is GPL-3.0 software that the user installs; MATSIM-AGENTS does not
implement its nuclear physics, bundle its source, or bundle its nuclear data.

Where the external pieces come from (full steps in ``docs/external/spectra-pka.md``):

* Executable: ``git clone https://github.com/fispact/SPECTRA-PKA.git`` into ``external/``, then
  a serial ``make``. Set ``SPECTRA_PKA_EXECUTABLE_PATH`` to the binary.
* PKA matrices (``nuclear_data_root``): the "Data download for SPECTRA-PKA" table on
  https://fispact.github.io/resources/nuclear_data_downloads/ links one Dropbox folder
  (``pkas.zip``, about 7.9 GB, TENDL-2015/2017/2019/2023). Site copies are listed per
  deployment (``deployments/``) with SOURCE.txt and SHA256SUMS. Extract the needed
  ``<El><AAA>s.asc`` files per campaign (docs/external/spectra-pka.md, section 2) and point
  ``PKA_DATA_DIRECTORY`` there.
* Neutron spectrum (``flux.file``): user-supplied. The upstream ``manual/test/fluxes_specter.dat``
  is a 709-group PWR spectrum for build checks only, not an application spectrum.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from matsim_agents.backends.external.spectra_pka.data_fetch import matrix_name
from matsim_agents.backends.external.spectra_pka.isotopes import IsotopeError, resolve_isotopes
from matsim_agents.core.yamlio import load_yaml_mapping

FRACTION_TOLERANCE = 1e-6
_SAFE_STUB = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ELEMENT = re.compile(r"^[A-Z][a-z]?$")


class ConfigError(ValueError):
    """The configuration is invalid or cannot be read."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(_Strict):
    """How the executable is launched. ``slurm`` goes through ``execution.slurm_jobs``."""

    mode: Literal["local", "slurm"] = "local"
    timeout_seconds: int = Field(default=3600, gt=0, le=7 * 24 * 3600)
    account: str | None = None
    qos: str | None = None
    partition: str | None = None
    time_limit: str | None = None

    @model_validator(mode="after")
    def _slurm_needs_account(self) -> ExecutionConfig:
        if self.mode == "slurm" and not self.account:
            raise ValueError("execution.account is required when execution.mode is 'slurm'")
        return self


class SpectraPKAConfig(_Strict):
    enabled: bool = False
    executable: Path | None = None
    installation_root: Path | None = None
    nuclear_data_root: Path | None = None
    workdir: Path = Path("outputs/external/spectra-pka")
    execution: ExecutionConfig = ExecutionConfig()
    # Raw outputs are never discarded; the field exists so a config that says otherwise
    # is rejected instead of silently ignored.
    preserve_raw_outputs: Literal[True] = True


class ExternalToolsConfig(_Strict):
    spectra_pka: SpectraPKAConfig = SpectraPKAConfig()


class ChannelSpec(_Strict):
    """One PKA matrix file (one target nuclide) and its atom fraction in the material."""

    pka_file: Path
    parent_ele: str
    parent_num: int = Field(gt=0, lt=400)
    atom_fraction: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _element(self) -> ChannelSpec:
        if not _ELEMENT.match(self.parent_ele):
            raise ValueError(f"parent_ele {self.parent_ele!r} is not an element symbol")
        return self


class FluxSpec(_Strict):
    """The neutron spectrum. Its normalisation is declared by the user, never inferred."""

    # Exactly one source: a native SPECTRA-PKA flux file, or a CSV/JSON group spectrum.
    file: Path | None = None
    spectrum_file: Path | None = None
    energy_units: Literal["eV", "keV", "MeV"] = "eV"
    flux_units: Literal["n/cm2/s", "n/m2/s"] = "n/cm2/s"
    value_kind: Literal["integrated", "per_energy", "per_lethargy"] = "integrated"
    absolute: bool = True
    identity: str = Field(min_length=1)
    normalization: str = Field(min_length=1)
    norm_type: Literal[1, 2, 3] = 2
    rescale_value: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _one_source(self) -> FluxSpec:
        if (self.file is None) == (self.spectrum_file is None):
            raise ValueError("give exactly one of flux.file and flux.spectrum_file")
        return self

    @property
    def source_path(self) -> Path:
        path = self.file or self.spectrum_file
        assert path is not None
        return path


class OptionsSpec(_Strict):
    do_global_sums: bool = True
    do_mtd_sums: bool = False
    do_exclude_light_from_total: bool = True
    energies_once_perfile: bool = True
    max_global_recoils: int = Field(default=200, gt=0, le=100000)


class CampaignSpec(_Strict):
    scenario_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    parent_alloy_id: str = Field(min_length=1)
    composition: dict[str, float]
    nuclear_data_identity: str = Field(min_length=1)
    # Either list channels, or set derive_channels to build them from the composition at load.
    channels: tuple[ChannelSpec, ...] = ()
    derive_channels: bool = False
    # element -> {mass number: fraction within the element}; the rest is natural abundance.
    isotope_overrides: dict[str, dict[int, float]] = {}
    flux: FluxSpec
    options: OptionsSpec = OptionsSpec()
    results_stub: str = "spectra_pka"

    @model_validator(mode="after")
    def _consistent(self) -> CampaignSpec:
        if not _SAFE_STUB.match(self.results_stub):
            raise ValueError("results_stub must match [A-Za-z0-9_.-]{1,64}")
        for element, fraction in self.composition.items():
            if not _ELEMENT.match(element):
                raise ValueError(f"composition key {element!r} is not an element symbol")
            if not math.isfinite(fraction) or fraction <= 0:
                raise ValueError(f"composition fraction of {element} must be positive")
        if abs(sum(self.composition.values()) - 1.0) > FRACTION_TOLERANCE:
            raise ValueError("composition atom fractions must sum to 1")
        if self.derive_channels:
            if self.channels:
                raise ValueError("derive_channels and an explicit channel list are exclusive")
            return self
        if self.isotope_overrides:
            raise ValueError("isotope_overrides only apply with derive_channels")
        if not self.channels:
            raise ValueError("give channels, or set derive_channels to build them")
        per_element: dict[str, float] = {}
        seen: set[tuple[str, int]] = set()
        for ch in self.channels:
            if ch.parent_ele not in self.composition:
                raise ValueError(f"channel element {ch.parent_ele} is not in the composition")
            key = (ch.parent_ele, ch.parent_num)
            if key in seen:
                raise ValueError(f"duplicate channel for {ch.parent_ele}{ch.parent_num}")
            seen.add(key)
            per_element[ch.parent_ele] = per_element.get(ch.parent_ele, 0.0) + ch.atom_fraction
        for element, fraction in self.composition.items():
            got = per_element.get(element)
            if got is None:
                raise ValueError(f"no channel covers composition element {element}")
            if abs(got - fraction) > FRACTION_TOLERANCE:
                raise ValueError(
                    f"isotope fractions of {element} sum to {got}, but the composition says "
                    f"{fraction}; SPECTRA-PKA silently renormalises ratios, so they must agree"
                )
        return self

    def isotopic_composition(self) -> dict[str, dict[int, float]]:
        """Isotope fractions within each element, derived from the channel atom fractions."""
        out: dict[str, dict[int, float]] = {}
        for ch in self.channels:
            out.setdefault(ch.parent_ele, {})[ch.parent_num] = (
                ch.atom_fraction / self.composition[ch.parent_ele]
            )
        return out


class ConfigFile(_Strict):
    external_tools: ExternalToolsConfig = ExternalToolsConfig()
    campaign: CampaignSpec | None = None


def _derive_channels(campaign: CampaignSpec, data_root: Path | None) -> CampaignSpec:
    """Replace derive_channels by explicit channels: natural or overridden isotopes per element."""
    if data_root is None:
        raise ConfigError("derive_channels needs external_tools.spectra_pka.nuclear_data_root")
    try:
        resolved = resolve_isotopes(
            campaign.composition, overrides=campaign.isotope_overrides or None
        )
    except IsotopeError as exc:
        raise ConfigError(str(exc)) from None
    channels = [
        {
            "pka_file": data_root / matrix_name(element, mass),
            "parent_ele": element,
            "parent_num": mass,
            "atom_fraction": fraction,
        }
        for (element, mass), fraction in sorted(resolved.isotope_fractions.items())
    ]
    data = campaign.model_dump()
    data.update(channels=channels, derive_channels=False, isotope_overrides={})
    try:
        return CampaignSpec(**data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from None


def _absolute(path: Path | None, base: Path) -> Path | None:
    if path is None:
        return None
    path = Path(os.path.expandvars(str(path)))  # unset ${VARS} stay literal and fail as missing
    return path if path.is_absolute() else (base / path).resolve()


def load_config_file(path: str | Path) -> ConfigFile:
    """Parse a YAML file. Relative paths resolve against the file's directory."""
    source = Path(path)
    try:
        data = load_yaml_mapping(source)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    except OSError as exc:
        raise ConfigError(f"{source}: {exc}") from None
    try:
        parsed = ConfigFile(**data)
    except (ValidationError, TypeError) as exc:
        raise ConfigError(f"{source}: {exc}") from None
    base = source.resolve().parent
    tool = parsed.external_tools.spectra_pka
    tool = tool.model_copy(
        update={
            "executable": _absolute(tool.executable, base),
            "installation_root": _absolute(tool.installation_root, base),
            "nuclear_data_root": _absolute(tool.nuclear_data_root, base),
            "workdir": _absolute(tool.workdir, base),
        }
    )
    campaign = parsed.campaign
    if campaign is not None and campaign.derive_channels:
        campaign = _derive_channels(campaign, tool.nuclear_data_root)
    if campaign is not None:
        campaign = campaign.model_copy(
            update={
                "flux": campaign.flux.model_copy(
                    update={
                        "file": _absolute(campaign.flux.file, base),
                        "spectrum_file": _absolute(campaign.flux.spectrum_file, base),
                    }
                ),
                "channels": tuple(
                    ch.model_copy(update={"pka_file": _absolute(ch.pka_file, base)})
                    for ch in campaign.channels
                ),
            }
        )
    return ConfigFile(external_tools=ExternalToolsConfig(spectra_pka=tool), campaign=campaign)


__all__ = [
    "CampaignSpec",
    "ChannelSpec",
    "ConfigError",
    "ConfigFile",
    "ExecutionConfig",
    "ExternalToolsConfig",
    "FluxSpec",
    "OptionsSpec",
    "SpectraPKAConfig",
    "load_config_file",
]
