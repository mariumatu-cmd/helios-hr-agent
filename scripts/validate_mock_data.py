"""Validate the referential and arithmetic integrity of the mock HR datasets.

    python scripts/validate_mock_data.py

The datasets are hand-authored to contain specific policy edge cases, which
makes them easy to break by hand too: a typo in an employee id, or a PTO
balance that no longer satisfies `carryover + accrued - used`, would quietly
turn a deliberate test case into a meaningless one. CI runs this on every push.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "mock_data"
TOLERANCE = 0.01

problems: list[str] = []


def fail(message: str) -> None:
    problems.append(message)


def load(name: str, key: str) -> list[dict]:
    payload = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    if "_meta" not in payload:
        fail(f"{name}.json has no _meta block")
    return payload[key]


def main() -> int:
    employees = load("employees", "employees")
    balances = load("pto_balances", "balances")
    elections = load("benefits_elections", "elections")
    history = load("international_work_history", "records")
    offices = load("offices", "offices")
    tickets = load("hr_tickets", "tickets")

    ids = {e["employee_id"] for e in employees}
    office_ids = {o["office_id"] for o in offices}

    if len(ids) != len(employees):
        fail("duplicate employee_id in employees.json")

    # -- referential integrity ---------------------------------------------
    for employee in employees:
        if employee["office_location"] not in office_ids:
            fail(f"{employee['employee_id']} references unknown office "
                 f"{employee['office_location']}")
        manager = employee.get("manager_id")
        if manager and manager not in ids:
            fail(f"{employee['employee_id']} references unknown manager {manager}")

    for label, rows in (("pto_balances", balances), ("benefits_elections", elections)):
        covered = {r["employee_id"] for r in rows}
        if covered - ids:
            fail(f"{label} references unknown employees: {sorted(covered - ids)}")
        if ids - covered:
            fail(f"{label} is missing employees: {sorted(ids - covered)}")

    for record in history:
        if record["employee_id"] not in ids:
            fail(f"international history {record['record_id']} references unknown employee")
        if record["end_date"] < record["start_date"]:
            fail(f"international history {record['record_id']} ends before it starts")

    for ticket in tickets:
        if ticket["employee_id"] not in ids:
            fail(f"ticket {ticket['ticket_id']} references unknown employee")

    # -- PTO arithmetic -----------------------------------------------------
    for balance in balances:
        if not balance["eligible"]:
            continue
        expected = (
            balance["carryover_hours"] + balance["accrued_ytd_hours"] - balance["used_ytd_hours"]
        )
        if abs(expected - balance["balance_hours"]) > TOLERANCE:
            fail(f"{balance['employee_id']}: balance_hours {balance['balance_hours']} != "
                 f"carryover + accrued - used ({expected:.2f})")

        available = balance["balance_hours"] - balance["scheduled_future_hours"]
        if abs(available - balance["available_hours"]) > TOLERANCE:
            fail(f"{balance['employee_id']}: available_hours {balance['available_hours']} != "
                 f"balance - scheduled ({available:.2f})")

        if balance["balance_hours"] > balance["accrual_cap_hours"] + TOLERANCE:
            fail(f"{balance['employee_id']}: balance exceeds the POL-PTO-001 §2.2 accrual cap")

    # -- eligibility consistency -------------------------------------------
    ineligible_types = {"temporary", "intern", "contractor"}
    by_id = {e["employee_id"]: e for e in employees}
    for balance in balances:
        expected = by_id[balance["employee_id"]]["employment_type"] not in ineligible_types
        if balance["eligible"] != expected:
            fail(f"{balance['employee_id']}: PTO eligibility contradicts employment type")
    for election in elections:
        expected = by_id[election["employee_id"]]["employment_type"] not in ineligible_types
        if election["benefits_eligible"] != expected:
            fail(f"{election['employee_id']}: benefits eligibility contradicts employment type")

    # -- the edge cases the evaluation suite depends on ---------------------
    required_cases = {
        "an employee with an expired international trip":
            any(r["employee_id"] == "E-1041" and r["end_date"] < "2025-09-12" for r in history),
        "an employee with too little PTO for a three-day request":
            any(b["employee_id"] == "E-1088" and b["available_hours"] < 24 for b in balances),
        "an employee under six months of service":
            any(e["hire_date"] > "2026-03-12" for e in employees),
        "an employee on a PIP":
            any(e.get("on_pip") for e in employees),
        "an employee on a work visa":
            any(str(e.get("work_authorization", "")).startswith("visa_") for e in employees),
        "a part-time employee":
            any(e["employment_type"] == "part_time" for e in employees),
        "a benefits-ineligible employment type":
            any(e["employment_type"] in ineligible_types for e in employees),
    }
    for description, present in required_cases.items():
        if not present:
            fail(f"the dataset no longer contains {description}")

    print(f"employees                  {len(employees)}")
    print(f"pto balances               {len(balances)}")
    print(f"benefits elections         {len(elections)}")
    print(f"international work records {len(history)}")
    print(f"offices                    {len(offices)}")
    print(f"seed tickets               {len(tickets)}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S) FOUND")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nMOCK DATA VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
