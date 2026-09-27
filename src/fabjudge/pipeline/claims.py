"""Evidence citation verification for LLM judge outputs."""

from __future__ import annotations

import math
from typing import Any


def verify_claims(packet: dict[str, Any], output: dict[str, Any], *,
                  abs_tolerance: float = 1e-6, rel_tolerance: float = 1e-6) -> dict:
    fields = packet["fields"]
    results = []
    for direction in ("evidence_for", "evidence_against"):
        for index, item in enumerate(output.get(direction, [])):
            field = item.get("field")
            cited = item.get("value")
            present = field in fields
            expected = fields.get(field)
            if present and isinstance(cited, (int, float)) and not isinstance(cited, bool):
                tolerance = abs_tolerance + rel_tolerance * abs(float(expected))
                matches = abs(float(cited) - float(expected)) <= tolerance
            else:
                tolerance, matches = 0.0, False
            results.append({
                "direction": direction, "index": index, "field": field,
                "cited_value": cited, "packet_value": expected if present else None,
                "matches_packet": bool(matches), "tolerance": float(tolerance),
                "issue": None if matches else "claim field/value does not match packet",
            })
    total = len(results)
    matching = sum(item["matches_packet"] for item in results)
    return {
        "claims": results,
        "claim_count": total,
        "matching_claim_count": matching,
        "mismatched_claim_count": total - matching,
        "agreement_rate": matching / total if total else None,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
    }
