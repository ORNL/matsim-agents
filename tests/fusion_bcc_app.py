"""The fusion 1,024-site BCC example application, loaded for tests."""

from pathlib import Path

from matsim_agents.core.application import load_application

APP_DIR = Path(__file__).resolve().parents[1] / "examples" / "fusion_bcc_1024"
APP = load_application(APP_DIR / "application.yaml")
CAMPAIGN = APP.resolve(APP.workflow_config)
TARGET = APP.target
COMPOSITION = dict(TARGET.composition)
REPEATS = tuple(TARGET.supercell)
SITES = TARGET.expected_sites
HOST = APP.parameters["transmutation"]["host"]
PRODUCTS = tuple(APP.parameters["transmutation"]["products"])
RADIATION_SPECIES = tuple(APP.parameters["radiation_species"])
