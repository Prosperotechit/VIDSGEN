"""The frontend's API contract files must match the backend exactly.

If this fails, run:  .venv/bin/python scripts/export_contract.py   (then: cd ../frontend && npm run gen:api)
"""

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_contract", Path(__file__).resolve().parents[1] / "scripts" / "export_contract.py")
export_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_contract)


def test_openapi_snapshot_is_current():
    committed = json.loads(export_contract.OPENAPI_PATH.read_text())
    assert committed == export_contract.build_openapi(), "backend/openapi.json is stale"


def test_frontend_constraints_are_current():
    committed = json.loads(export_contract.CONSTRAINTS_PATH.read_text())
    assert committed == export_contract.build_constraints(), "frontend/src/api/constraints.json is stale"


def test_constraints_values_match_validation():
    c = export_contract.build_constraints()
    assert [r["value"] for r in c["resolutions"]] == ["1280x704", "704x1280", "832x480", "480x832", "640x640"]
    assert c["numFrames"] == {"min": 17, "max": 121, "step": 4, "default": 49}
    assert c["fps"] == {"default": 24, "min": 8, "max": 30}
    assert c["steps"] == {"default": 30, "min": 1, "max": 60}
    assert c["guidanceScale"] == {"default": 5.0, "min": 1.0, "max": 15.0}
    assert c["image"]["mimeTypes"] == ["image/jpeg", "image/png", "image/webp"]
