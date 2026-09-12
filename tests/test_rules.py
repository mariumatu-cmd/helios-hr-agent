"""Policy rule engine.

Each test names the corpus clause it encodes. These are the calculations the
LLM is explicitly forbidden from doing itself, so they have to be right here.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest

from mcp_server import data

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- resolution ------------------------------------------------------------
@pytest.mark.parametrize("identifier", ["E-1041", "maya.rodriguez@helios.example.com",
                                        "Maya Rodriguez", "maya"])
def test_employee_resolves_by_several_identifiers(identifier):
    assert data.resolve_employee(identifier)["employee_id"] == "E-1041"


def test_unknown_employee_raises():
    with pytest.raises(LookupError):
        data.employee_profile("Nobody McNotreal")


def test_ambiguous_partial_name_refuses_to_guess():
    """Silently picking one of several matches would attach a correct-looking
    answer to the wrong person."""
    people = data.employees()
    first_names = [e["full_name"].split()[0] for e in people]
    ambiguous = next((n for n in first_names if first_names.count(n) > 1), None)
    if ambiguous is None:
        pytest.skip("no ambiguous first name in the dataset")
    with pytest.raises(LookupError):
        data.resolve_employee(ambiguous)


# --- POL-INTL-001 §2: rolling 12-month window ------------------------------
def test_rolling_window_excludes_expired_trips():
    """Maya has two trips on file; one ended before the window opened. Summing
    all history gives 24 days, which is the wrong answer."""
    usage = data.international_usage("E-1041")
    assert usage["days_used"] == 12
    assert len(usage["excluded_records"]) == 1
    assert "before the rolling window opened" in usage["excluded_records"][0]["excluded_because"]


def test_rolling_window_reports_its_own_bounds():
    usage = data.international_usage("E-1041")
    start = dt.date.fromisoformat(usage["window_start"])
    end = dt.date.fromisoformat(usage["window_end"])
    assert (end - start).days == data.INTL_ROLLING_WINDOW_DAYS


def test_six_week_request_breaches_the_thirty_day_limit():
    result = data.check_international_request("E-1041", "Portugal", "2026-10-05", "2026-11-15")
    limit = next(f for f in result["findings"] if f["check"] == "rolling_limit")
    assert not result["compliant"]
    assert limit["requested_days"] == 42
    assert limit["projected_total_days"] == 54
    assert limit["overage_days"] == 24
    assert data.INTL_LIMIT_CITATION in result["citations"]


def test_short_trip_within_the_limit_is_compliant():
    result = data.check_international_request("E-1041", "Portugal", "2026-10-05", "2026-10-18")
    assert result["compliant"]


def test_visa_holder_triggers_immigration_review():
    result = data.check_international_request("E-1120", "Brazil", "2026-10-01", "2026-10-10")
    immigration = next(f for f in result["findings"] if f["check"] == "immigration")
    assert immigration["status"] == "review_required"
    assert result["requires_review"]


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError):
        data.check_international_request("E-1041", "Spain", "2026-10-10", "2026-10-01")


# --- POL-PTO-001 §3, §4 ----------------------------------------------------
@pytest.mark.parametrize("days,expected", [(1, 3), (2, 3), (3, 10), (5, 10), (7, 20), (15, 30)])
def test_notice_bands_match_the_policy_table(days, expected):
    assert data.required_notice_days(days)[0] == expected


def test_insufficient_balance_blocks_the_request():
    result = data.check_pto_request("E-1088", "2026-12-01", 3)
    balance = next(f for f in result["findings"] if f["check"] == "balance")
    assert balance["status"] == "fail"
    assert balance["shortfall_hours"] > 0


def test_short_notice_blocks_the_request():
    result = data.check_pto_request("E-1041", "2026-09-16", 3)
    notice = next(f for f in result["findings"] if f["check"] == "notice")
    assert notice["status"] == "fail"
    assert notice["given_business_days"] < notice["required_business_days"]


def test_compliant_pto_request_passes_every_check():
    result = data.check_pto_request("E-1041", "2026-10-19", 3)
    assert result["compliant"], result["blocking_reasons"]


def test_conference_week_is_a_blackout():
    result = data.check_pto_request("E-1041", "2026-09-14", 3)
    blackout = next(f for f in result["findings"] if f["check"] == "blackout")
    assert blackout["status"] == "fail"
    assert blackout["exception_path"]


def test_blackout_does_not_apply_to_short_absences():
    """POL-PTO-001 §4 restricts absences of three or more consecutive days only."""
    result = data.check_pto_request("E-1041", "2026-09-14", 1)
    blackout = next(f for f in result["findings"] if f["check"] == "blackout")
    assert blackout["status"] == "pass"


def test_fiscal_close_windows_are_computed_not_hardcoded():
    windows = data.fiscal_close_windows(2026)
    assert len(windows) == 4
    for start, end, _ in windows:
        assert start < end
        assert start.weekday() < 5 and end.weekday() < 5


def test_customer_support_peak_season_blackout():
    result = data.check_pto_request("E-1088", "2026-11-20", 3)
    blackout = next(f for f in result["findings"] if f["check"] == "blackout")
    assert blackout["status"] == "fail"
    assert any("Peak support" in c["window"] for c in blackout["conflicts"])


def test_zero_days_is_rejected():
    with pytest.raises(ValueError):
        data.check_pto_request("E-1041", "2026-10-19", 0)


def test_bad_date_format_is_rejected():
    with pytest.raises(ValueError):
        data.check_pto_request("E-1041", "19 October", 3)


def test_business_days_skips_weekends():
    assert data.business_days_between(dt.date(2026, 9, 11), dt.date(2026, 9, 14)) == 1


# --- POL-REMOTE-001 / POL-PERF-001 -----------------------------------------
def test_short_tenure_blocks_remote_change():
    result = data.check_remote_arrangement_change("E-1120", "remote")
    assert not result["compliant"]
    assert data.REMOTE_TENURE_CITATION in result["citations"]


def test_active_pip_blocks_fully_remote():
    result = data.check_remote_arrangement_change("E-1055", "remote")
    assert not result["compliant"]
    assert any("Performance Improvement Plan" in r for r in result["blocking_reasons"])


def test_pip_does_not_block_a_hybrid_change():
    """POL-PERF-001 §8.3 restricts fully-remote changes specifically."""
    result = data.check_remote_arrangement_change("E-1055", "hybrid")
    assert all(f["check"] != "performance" for f in result["findings"])


def test_invalid_arrangement_is_rejected():
    with pytest.raises(ValueError):
        data.check_remote_arrangement_change("E-1041", "lunar")


# --- POL-BEN-001 -----------------------------------------------------------
def test_part_time_has_no_disability_cover():
    benefits = data.benefits_status("E-1099")
    assert benefits["benefits_eligible"]
    assert benefits["short_term_disability"] is False


@pytest.mark.parametrize("employee_id", ["E-1150", "E-1177", "C-2003"])
def test_ineligible_employment_types_are_flagged(employee_id):
    benefits = data.benefits_status(employee_id)
    assert benefits["benefits_eligible"] is False
    assert "POL-BEN-001" in benefits["note"]


def test_ineligible_employee_has_no_pto_accrual():
    balance = data.pto_balance("E-1177")
    assert balance["eligible"] is False
    assert "POL-BEN-001" in balance["note"]


# --- mock writes -----------------------------------------------------------
def test_created_ticket_is_never_persisted():
    ticket = data.create_ticket("E-1041", "pto", "Test", "Body")
    on_disk = json.loads((ROOT / "mock_data" / "hr_tickets.json").read_text(encoding="utf-8"))
    assert ticket["persisted"] is False
    assert all(t["ticket_id"] != ticket["ticket_id"] for t in on_disk["tickets"])


def test_tickets_are_routed_by_category():
    assert data.create_ticket(
        "E-1041", "international_remote_work", "s", "b"
    )["assigned_team"] == "Global Mobility"
    assert data.create_ticket("E-1041", "benefits", "s", "b")["assigned_team"] == "Benefits Team"


def test_drafted_email_is_never_sent():
    draft = data.draft_email("E-1041", "Subject", "Body")
    assert draft["sent"] is False
    assert draft["to"].endswith("@helios.example.com")


def test_empty_subject_is_rejected():
    with pytest.raises(ValueError):
        data.create_ticket("E-1041", "pto", "   ", "body")


# --- determinism -----------------------------------------------------------
def test_as_of_date_is_pinned_not_wall_clock():
    """If this ever reads the real clock, every dated expectation in the
    evaluation suite starts drifting."""
    assert data.today() == dt.date.fromisoformat("2026-09-12")


def test_repeated_calls_are_identical():
    first = data.check_pto_request("E-1041", "2026-10-19", 3)
    second = data.check_pto_request("E-1041", "2026-10-19", 3)
    assert first == second
