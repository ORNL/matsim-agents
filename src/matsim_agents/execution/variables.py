"""Shell-style ``${VAR}`` substitution shared by YAML configuration loaders.

Supported forms:

* ``${VAR}``                  — required; raises if unset.
* ``${VAR:-default}``          — falls back to ``default`` if unset.
* ``${VAR:?error message}``    — raises with that message if unset.

Values come from ``os.environ`` first, then from an optional top-level
``vars:`` mapping in the YAML document.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:[-?][^}]*)?\}")
_MAX_PASSES = 10


def variable_defaults(vars_block: Any, *, source: object) -> dict[str, str]:
    """Return fallback values from a YAML ``vars:`` mapping.

    A ``vars:`` entry may reference the environment with its own name as a
    fallback default, e.g. ``RUNS_ROOT: ${RUNS_ROOT:-/some/path}`` ("use
    $RUNS_ROOT if exported, else this literal"). That self-reference is
    resolved here; otherwise expansion would look the name up in the defaults
    and loop back to the entry's own raw string. References to *other* vars
    are left for :func:`expand_variables`.
    """

    if not isinstance(vars_block, dict):
        return {}

    def _break_self_ref(key: str, value: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            modifier = match.group(2) or ""
            if os.environ.get(name):
                return os.environ[name]
            if name == key:
                if modifier.startswith(":-"):
                    return modifier[2:]
                if modifier.startswith(":?"):
                    msg = modifier[2:].strip() or f"required variable {name!r} is unset"
                    raise ValueError(f"{source}: {msg}")
            return match.group(0)

        return VARIABLE_PATTERN.sub(_sub, value)

    return {str(k): _break_self_ref(str(k), str(v)) for k, v in vars_block.items()}


def expand_variables(text: str, defaults: Mapping[str, str], *, source: object) -> str:
    """Expand every ``${VAR}`` reference in *text*, iterating so vars can nest."""

    def _resolve(match: re.Match[str]) -> str:
        name = match.group(1)
        modifier = match.group(2) or ""
        if os.environ.get(name):
            return os.environ[name]
        if name in defaults:
            return defaults[name]
        if modifier.startswith(":-"):
            return modifier[2:]
        if modifier.startswith(":?"):
            msg = modifier[2:].strip() or f"required variable {name!r} is unset"
            raise ValueError(f"{source}: {msg}")
        raise ValueError(
            f"{source}: undefined variable ${{{name}}} "
            "(set it in the environment, in the YAML 'vars:' block, "
            "or use ${VAR:-default} syntax)."
        )

    expanded = text
    for _ in range(_MAX_PASSES):
        new_text = VARIABLE_PATTERN.sub(_resolve, expanded)
        if new_text == expanded:
            return expanded
        expanded = new_text
    raise ValueError(
        f"{source}: variable substitution did not converge after {_MAX_PASSES} passes "
        "(possible circular reference in 'vars:')."
    )


__all__ = ["VARIABLE_PATTERN", "expand_variables", "variable_defaults"]
