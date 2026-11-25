"""SYNTHETIC candidates A/B/C for exercising the fusion scoring pipeline (F12).

Every number here is invented to exercise code paths. None is a material
property, a benchmark, or a prediction, and every record says so.
"""

from __future__ import annotations

from matsim_agents.core.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.fusion.scenario import EvaluationScenario, EvidenceContext, FusionEvidence
from matsim_agents.fusion.scoring import BenchmarkSet
from matsim_agents.science.properties.calculators import PropertyEvidence, PropertyStatus

SYNTHETIC_LABEL = "SYNTHETIC demonstration data - not a material property"

SCENARIO = EvaluationScenario(
    scenario_id="synthetic-divertor-1200K",
    description="SYNTHETIC divertor-like state for pipeline tests",
    temperature_K=1200.0,
    irradiation="synthetic-fw",
    neutron_spectrum="synthetic-spectrum",
    cooling_time_s=3.15e7,
    plasma_ion="D",
    plasma_ion_energy_eV=100.0,
)

BENCHMARKS = BenchmarkSet.model_validate(
    {
        "benchmark_version": "synthetic-demo-v1",
        "provenance": "invented anchors for software tests",
        "synthetic": True,
        "objectives": {
            "radiation_damage_recovery": {
                "observables": {
                    "defect_recovery_fraction": {"worst": 0.0, "target": 1.0, "source": "syn"}
                }
            },
            "thermal_conductivity": {
                "observables": {
                    "thermal_conductivity_total": {"worst": 20.0, "target": 170.0, "source": "syn"}
                }
            },
            "mechanical_integrity": {
                "observables": {
                    "yield_strength": {"worst": 200.0, "target": 1200.0, "source": "syn"},
                    "fracture_toughness": {"worst": 5.0, "target": 50.0, "source": "syn"},
                }
            },
            "thermal_shock_fatigue": {
                "observables": {
                    "thermal_shock_damage_threshold": {
                        "worst": 100.0,
                        "target": 1000.0,
                        "source": "syn",
                    }
                }
            },
            "plasma_erosion_resistance": {
                "observables": {
                    "sputtering_yield": {"worst": 1e-2, "target": 1e-4, "source": "syn"}
                }
            },
            "high_temperature_stability": {
                "observables": {
                    "max_phase_stable_temperature": {
                        "worst": 1000.0,
                        "target": 2000.0,
                        "source": "syn",
                    }
                }
            },
            "low_hydrogen_retention": {
                "observables": {
                    "hydrogen_retention_fraction_T": {
                        "worst": 0.1,
                        "target": 0.001,
                        "source": "syn",
                    }
                }
            },
            "low_activation_shutdown_dose": {
                "observables": {
                    "shutdown_dose_rate": {"worst": 1e-1, "target": 1e-5, "source": "syn"}
                }
            },
        },
    }
)

# (value, one-sigma) per observable; C deliberately lacks a shutdown dose rate.
_VALUES: dict[str, dict[str, tuple[float, float]]] = {
    "A": {
        "defect_recovery_fraction": (0.70, 0.05),
        "thermal_conductivity_total": (120.0, 10.0),
        "yield_strength": (900.0, 50.0),
        "fracture_toughness": (30.0, 3.0),
        "thermal_shock_damage_threshold": (600.0, 50.0),
        "sputtering_yield": (1e-3, 2e-4),
        "max_phase_stable_temperature": (1800.0, 50.0),
        "hydrogen_retention_fraction_T": (0.01, 0.002),
        "shutdown_dose_rate": (1e-3, 2e-4),
    },
    "B": {
        "defect_recovery_fraction": (0.50, 0.05),
        "thermal_conductivity_total": (110.0, 10.0),
        "yield_strength": (700.0, 50.0),
        "fracture_toughness": (20.0, 3.0),
        "thermal_shock_damage_threshold": (500.0, 50.0),
        "sputtering_yield": (2e-3, 3e-4),
        "max_phase_stable_temperature": (1600.0, 50.0),
        "hydrogen_retention_fraction_T": (0.02, 0.004),
        "shutdown_dose_rate": (2e-2, 4e-3),
    },
    "C": {
        "defect_recovery_fraction": (0.80, 0.05),
        "thermal_conductivity_total": (115.0, 10.0),
        "yield_strength": (1000.0, 50.0),
        "fracture_toughness": (35.0, 3.0),
        "thermal_shock_damage_threshold": (700.0, 50.0),
        "sputtering_yield": (8e-4, 1e-4),
        "max_phase_stable_temperature": (1900.0, 50.0),
        "hydrogen_retention_fraction_T": (0.008, 0.002),
    },
}


def _record(candidate: str, prop: str, value: float, sigma: float) -> FusionEvidence:
    ev = PropertyEvidence(
        structure_id=f"synthetic-{candidate}",
        composition=f"SYNTHETIC-{candidate}",
        property=prop,
        status=PropertyStatus.SURROGATE_PREDICTED,
        value=value,
        uncertainty=sigma,
        unit=PROPERTY_UNITS[prop],
        conditions={
            "temperature_K": SCENARIO.temperature_K,
            "irradiation": SCENARIO.irradiation,
        },
        method="synthetic",
        backend="synthetic",
        evidence_ids=[f"synthetic-{candidate}-{prop}"],
        evidence_level=EvidenceLevel.HYPOTHESIS,
        limitations=[SYNTHETIC_LABEL],
    )
    ctx = EvidenceContext(
        neutron_spectrum=SCENARIO.neutron_spectrum,
        cooling_time_s=SCENARIO.cooling_time_s,
        plasma_ion=SCENARIO.plasma_ion,
        plasma_ion_energy_eV=SCENARIO.plasma_ion_energy_eV,
    )
    return FusionEvidence(evidence=ev, context=ctx)


def candidates() -> dict[str, list[FusionEvidence]]:
    return {
        cid: [_record(cid, p, v, s) for p, (v, s) in vals.items()] for cid, vals in _VALUES.items()
    }


__all__ = ["BENCHMARKS", "SCENARIO", "SYNTHETIC_LABEL", "candidates"]
