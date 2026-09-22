"""Pure, side-effect-free tool implementations the agent can call.

Every tool here takes plain JSON-serializable arguments and returns a
plain JSON-serializable dict -- there is no hidden state and no network
call, which is what makes them straightforward to unit test directly
(``tests/test_tools.py``) and safe to replay from the audit trail.

All data is synthetic and lives under ``data/`` as JSON / text files
shipped with the package; nothing here reads or writes real patient
information.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"


class ToolError(Exception):
    """Raised when a tool cannot fulfil its request (e.g. unknown id).

    The agent loop catches this and feeds the message back to the model
    as a tool result, the same way a real API would return a 404/422,
    rather than letting the whole run crash on a bad argument.
    """


@lru_cache(maxsize=1)
def _load_policies() -> dict[str, Any]:
    with open(DATA_DIR / "policies.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_patients() -> dict[str, Any]:
    with open(DATA_DIR / "patients.json", encoding="utf-8") as f:
        return json.load(f)


def lookup_policy(policy_id: str) -> dict[str, Any]:
    """Return the structured coverage rules for a policy.

    Includes the plan name, deductible/coinsurance figures, the flat
    covered-procedures table, and which procedure codes require prior
    authorization. Does NOT include free-text exclusions or conditional
    rules -- those only live in the policy documents and must be found
    via ``search_policy_documents``.
    """
    policies = _load_policies()
    if policy_id not in policies:
        raise ToolError(f"No policy found with id '{policy_id}'.")
    return policies[policy_id]


def check_prior_claims(patient_id: str) -> dict[str, Any]:
    """Return a patient's prior claims history, for duplicate/repeat checks."""
    patients = _load_patients()
    if patient_id not in patients:
        raise ToolError(f"No patient found with id '{patient_id}'.")
    record = patients[patient_id]
    return {
        "patient_id": patient_id,
        "policy_id": record["policy_id"],
        "prior_claims": record["prior_claims"],
    }


def calculate_coverage(
    policy_id: str, procedure_code: str, billed_amount: float
) -> dict[str, Any]:
    """Compute what the plan and the patient each owe for a billed amount.

    Applies the remaining deductible first, then coinsurance on the
    remainder, capped so the patient never pays past their remaining
    out-of-pocket maximum. This mirrors the standard (simplified) US
    health-plan cost-share waterfall: deductible -> coinsurance -> OOP max.
    """
    policy = lookup_policy(policy_id)
    if procedure_code not in policy["covered_procedures"]:
        return {
            "policy_id": policy_id,
            "procedure_code": procedure_code,
            "billed_amount": billed_amount,
            "is_covered_procedure": False,
            "patient_owes": billed_amount,
            "plan_pays": 0.0,
            "note": "Procedure code is not on this plan's covered-procedures list.",
        }

    remaining_deductible = max(
        0.0, policy["deductible"] - policy["deductible_met"]
    )
    amount_after_deductible = max(0.0, billed_amount - remaining_deductible)
    deductible_applied = min(billed_amount, remaining_deductible)

    coinsurance_owed = amount_after_deductible * policy["coinsurance_rate"]
    patient_owes_raw = deductible_applied + coinsurance_owed

    remaining_oop = max(
        0.0, policy["out_of_pocket_max"] - policy["out_of_pocket_met"]
    )
    patient_owes = min(patient_owes_raw, remaining_oop)
    plan_pays = billed_amount - patient_owes

    return {
        "policy_id": policy_id,
        "procedure_code": procedure_code,
        "billed_amount": round(billed_amount, 2),
        "is_covered_procedure": True,
        "deductible_applied": round(deductible_applied, 2),
        "coinsurance_owed": round(coinsurance_owed, 2),
        "patient_owes": round(patient_owes, 2),
        "plan_pays": round(plan_pays, 2),
        "out_of_pocket_max_reached": patient_owes_raw > remaining_oop,
    }
