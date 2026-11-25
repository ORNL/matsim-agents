"""Structural validation of parsed SPECTRA-PKA output.

"Structurally valid" means the file is complete and self-consistent. It does not mean the PKA
rates are correct: that needs a real run compared with an independent reference, which this
module cannot supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matsim_agents.integrations.external.spectra_pka.config import CampaignSpec
from matsim_agents.integrations.external.spectra_pka.parser import BlockKind, ParsedOutput
from matsim_agents.science.irradiation.contracts import ValidationStatus

FLUX_SUM_RTOL = 1e-3
NORM_SUM_SLACK = 1e-6
# Global matrices write ``cumulative = SUM(1:j) / SUM(all)`` (global_sums.f90) as ES11.4, so the
# last row is 1 to four decimals. Anything else means the matrix was truncated.
CUMULATIVE_END_ATOL = 1e-3
_GLOBAL_KINDS = frozenset(
    {
        BlockKind.GLOBAL_ISOTOPE,
        BlockKind.GLOBAL_ELEMENT,
        BlockKind.GLOBAL_TOTAL,
        BlockKind.GLOBAL_MT,
    }
)


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status is ValidationStatus.STRUCTURALLY_VALID

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "meaning": "structural completeness only; not a physics validation",
        }


def validate_output(
    parsed: ParsedOutput,
    campaign: CampaignSpec | None = None,
    *,
    end_marker_seen: bool | None = None,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if end_marker_seen is False:
        errors.append("the 'END as expected' marker is missing: the run did not finish normally")
    elif end_marker_seen is None:
        warnings.append("the end-of-run marker was not checked")

    inputs = parsed.of_kind(BlockKind.INPUT_SPECTRUM)
    channels = parsed.of_kind(BlockKind.CHANNEL)
    if not inputs:
        errors.append("no input-spectrum block")
    if not channels:
        errors.append("no channel blocks")
    if not parsed.of_kind(BlockKind.GLOBAL_ISOTOPE) and not parsed.of_kind(
        BlockKind.GLOBAL_ELEMENT
    ):
        warnings.append("no global recoil matrices (do_global_sums off?); cannot build a spectrum")

    for block in parsed.blocks:
        previous_high = None
        for row in block.rows:
            lo, hi = row[0], row[1]
            if lo < 0 or hi <= lo:
                errors.append(f"{block.title[:50]!r}: bad energy bin ({lo}, {hi})")
                break
            if previous_high is not None and lo < previous_high * (1 - 1e-9):
                errors.append(f"{block.title[:50]!r}: bins overlap or are not ascending")
                break
            previous_high = hi
        for row in block.rows:
            if any((not math.isfinite(v)) or v < 0 for v in row[2:]):
                errors.append(f"{block.title[:50]!r}: negative or non-finite value in a row")
                break
        if block.kind is not BlockKind.INPUT_SPECTRUM and block.rows:
            norm_col = [r[3] for r in block.rows]
            if any(v > 1.0 + NORM_SUM_SLACK for v in norm_col):
                warnings.append(f"{block.title[:50]!r}: norm_sum exceeds 1")
        if block.kind in _GLOBAL_KINDS and block.rows:
            final = block.rows[-1][4]
            if abs(final - 1.0) > CUMULATIVE_END_ATOL:
                errors.append(
                    f"{block.title[:50]!r}: final cumulative fraction is {final:.6g}, not 1; "
                    "the matrix is truncated"
                )
        if block.kind is BlockKind.INPUT_SPECTRUM and "total_flux" in block.footer:
            total = math.fsum(r[2] for r in block.rows)
            stated = block.footer["total_flux"]
            if stated > 0 and abs(total - stated) > FLUX_SUM_RTOL * stated:
                warnings.append(
                    f"input spectrum rows sum to {total:.6g} but TOTAL FLUX says {stated:.6g}"
                )

    if all(not b.rows for b in channels) and channels:
        errors.append("every channel block is empty: the run produced no PKAs")

    if campaign is not None:
        present = {(b.parent_element, b.parent_mass_number) for b in channels}
        for ch in campaign.channels:
            if (ch.parent_ele, ch.parent_num) not in present:
                warnings.append(
                    f"no channel block for {ch.parent_ele}{ch.parent_num}; it may produce no "
                    "PKAs in this spectrum or the run was incomplete"
                )
        if parsed.indexes and len(parsed.indexes) < len(channels):
            warnings.append("the indexes file lists fewer entries than there are channel blocks")

    status = (
        ValidationStatus.STRUCTURALLY_INVALID if errors else ValidationStatus.STRUCTURALLY_VALID
    )
    return ValidationReport(status, tuple(errors), tuple(warnings))


__all__ = ["ValidationReport", "validate_output"]
