"""Spot-check the rule engine against the corpus edge cases.

    python scripts/check_rules.py

These are the cases the corpus and mock data were designed around; if any of
them regress, the evaluation suite is measuring the wrong system.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server import data  # noqa: E402


def show(label: str, payload) -> None:
    print(f"\n=== {label}")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2600])


def main() -> int:
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        print(("  PASS  " if condition else "  FAIL  ") + message)
        if not condition:
            failures.append(message)

    usage = data.international_usage("E-1041")
    show("Maya Rodriguez rolling international usage", {
        k: usage[k] for k in ("window_start", "window_end", "days_used", "days_remaining")
    } | {"counted": [r["record_id"] for r in usage["counted_records"]],
         "excluded": [(r["record_id"], r["excluded_because"]) for r in usage["excluded_records"]]})
    expect(usage["days_used"] == 12, "Maya has used 12 days, not 24 (Spain trip aged out of window)")

    intl = data.check_international_request("Maya", "Portugal", "2026-10-05", "2026-11-15")
    show("Maya 6-week Portugal request", {
        "compliant": intl["compliant"],
        "blocking_reasons": intl["blocking_reasons"],
        "citations": intl["citations"],
    })
    expect(not intl["compliant"], "42-day Portugal request breaches the 30-day rolling limit")
    expect(intl["findings"][0]["overage_days"] == 54 - 30,
           f"overage computed as {intl['findings'][0]['overage_days']} days")

    pto = data.check_pto_request("E-1088", "2026-09-21", 3)
    show("Jonas Weber 3-day request next week", {
        "compliant": pto["compliant"],
        "blocking_reasons": pto["blocking_reasons"],
    })
    expect(not pto["compliant"], "Jonas cannot take 3 days (insufficient balance and short notice)")
    balance_finding = next(f for f in pto["findings"] if f["check"] == "balance")
    expect(balance_finding["status"] == "fail", "balance check fails for Jonas")
    notice_finding = next(f for f in pto["findings"] if f["check"] == "notice")
    expect(notice_finding["status"] == "fail",
           f"notice check fails ({notice_finding['given_business_days']} given, "
           f"{notice_finding['required_business_days']} required)")

    conf = data.check_pto_request("E-1041", "2026-09-14", 3)
    blackout = next(f for f in conf["findings"] if f["check"] == "blackout")
    show("Conference-week request", {"status": blackout["status"], "detail": blackout["detail"]})
    expect(blackout["status"] == "fail", "Helios Forum conference week is a blackout window")

    single = data.check_pto_request("E-1041", "2026-09-14", 1)
    single_blackout = next(f for f in single["findings"] if f["check"] == "blackout")
    expect(single_blackout["status"] == "pass",
           "a 1-day request during conference week is not blocked (policy covers 3+ days)")

    remote = data.check_remote_arrangement_change("E-1120", "remote")
    show("Tomas Silva remote change", {
        "compliant": remote["compliant"], "blocking_reasons": remote["blocking_reasons"]})
    expect(not remote["compliant"], "Tomas fails the 6-month tenure requirement")

    pip_case = data.check_remote_arrangement_change("E-1055", "remote")
    show("Marcus Doyle remote change", {
        "compliant": pip_case["compliant"], "blocking_reasons": pip_case["blocking_reasons"]})
    expect(not pip_case["compliant"], "an active PIP blocks a fully-remote change")

    visa = data.check_international_request("E-1120", "Brazil", "2026-10-01", "2026-10-10")
    immigration = next(f for f in visa["findings"] if f["check"] == "immigration")
    expect(immigration["status"] == "review_required",
           "H-1B holder triggers Immigration Counsel review")

    benefits = data.benefits_status("E-1099")
    show("Sofia Marino benefits", {k: benefits[k] for k in
                                   ("benefits_eligible", "short_term_disability",
                                    "long_term_disability", "employment_type")})
    expect(benefits["short_term_disability"] is False, "part-time employee has no STD")

    ticket = data.create_ticket("E-1041", "international_remote_work", "Portugal request",
                                "Requesting 6 weeks from Lisbon.")
    expect(ticket["persisted"] is False and ticket["assigned_team"] == "Global Mobility",
           "mock ticket is in-memory and routed to Global Mobility")
    seed_file = json.loads((ROOT / "mock_data" / "hr_tickets.json").read_text(encoding="utf-8"))
    expect(all(t["ticket_id"] != ticket["ticket_id"] for t in seed_file["tickets"]),
           "created ticket was not written to hr_tickets.json")

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
