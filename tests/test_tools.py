import pytest

from claims_triage_agent.tools import (
    ToolError,
    calculate_coverage,
    check_prior_claims,
    lookup_policy,
)


def test_lookup_policy_returns_known_policy():
    policy = lookup_policy("POL-1001")
    assert policy["plan_name"] == "Meridian Health Gold PPO"
    assert policy["covered_procedures"]["29881"] is True


def test_lookup_policy_unknown_id_raises():
    with pytest.raises(ToolError):
        lookup_policy("POL-DOES-NOT-EXIST")


def test_check_prior_claims_returns_history():
    result = check_prior_claims("PAT-001")
    assert result["policy_id"] == "POL-1001"
    assert len(result["prior_claims"]) == 1
    assert result["prior_claims"][0]["procedure_code"] == "29881"


def test_check_prior_claims_unknown_patient_raises():
    with pytest.raises(ToolError):
        check_prior_claims("PAT-DOES-NOT-EXIST")


def test_calculate_coverage_applies_remaining_deductible_then_coinsurance():
    # POL-1001: deductible 1000, deductible_met 400 -> remaining 600
    # coinsurance 20%. Billed 5200 -> 600 deductible + 20% of 4600 = 920
    result = calculate_coverage("POL-1001", "29881", 5200.0)
    assert result["is_covered_procedure"] is True
    assert result["deductible_applied"] == 600.0
    assert result["coinsurance_owed"] == 920.0
    assert result["patient_owes"] == 1520.0
    assert result["plan_pays"] == 3680.0


def test_calculate_coverage_not_covered_procedure():
    result = calculate_coverage("POL-1001", "00000", 100.0)
    assert result["is_covered_procedure"] is False
    assert result["patient_owes"] == 100.0
    assert result["plan_pays"] == 0.0


def test_calculate_coverage_respects_out_of_pocket_max():
    # POL-2002: deductible already fully met (3000/3000), OOP max 8000,
    # OOP met 3000 -> remaining OOP room is 5000. A huge bill should cap
    # patient responsibility at that remaining OOP room.
    result = calculate_coverage("POL-2002", "70551", 50000.0)
    assert result["patient_owes"] <= 5000.0
    assert result["out_of_pocket_max_reached"] is True
