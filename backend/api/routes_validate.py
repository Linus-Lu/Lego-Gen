"""Validation endpoint for LEGO build descriptions."""

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Body

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.inference.stability_checker import StabilityChecker

router = APIRouter(prefix="/api", tags=["validate"])

_checker = StabilityChecker()


@router.post("/validate")
async def validate_build(body: dict = Body(...)):
    """Validate a LEGO build description for stability and legality."""
    report = _checker.validate(body)
    return asdict(report)
