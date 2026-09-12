"""Mock HR system of record, and the policy rules that read it.

This module is the "structured data" half of the system: the RAG corpus says
what the rules *are*, and this says what is *true of a given employee*. Keeping
the rule evaluation here -- rather than in the MCP tool wrappers or in a prompt
-- means the arithmetic that decides whether a request is compliant is
deterministic, unit-testable, and identical every run. The LLM is left to do
what it is good at (reading the policy, explaining, drafting) and is never
asked to add up dates.

Everything here is synthetic. Writes are held in memory only: a created ticket
is visible for the life of the process and is never persisted to
``mock_data/hr_tickets.json``, so no demo run can corrupt the seed state or
produce an irreversible side effect.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import threading
from typing import Any

from config import settings

# ---------------------------------------------------------------------------
# Policy constants. Each is traceable to a clause in the corpus; the citation is
# returned alongside every decision so the agent can quote the source rather
# than assert the number.
# ---------------------------------------------------------------------------
INTL_ROLLING_WINDOW_DAYS = 365
INTL_ANNUAL_LIMIT_DAYS = 30
INTL_LIMIT_CITATION = "POL-INTL-001 §2 Rolling 30-Day Limit"
INTL_IMMIGRATION_CITATION = "POL-INTL-001 §4.4 Immigration Review"

# (max consecutive business days, minimum business days of notice)
PTO_NOTICE_BANDS = ((2, 3), (5, 10), (10, 20), (10_000, 30))
PTO_NOTICE_CITATION = "POL-PTO-001 §3.1 Notice Requirements"
PTO_BALANCE_CITATION = "POL-PTO-001 §3.3 Insufficient Balance"
PTO_BLACKOUT_CITATION = "POL-PTO-001 §4 Blackout Periods"

REMOTE_TENURE_MONTHS = 6
REMOTE_TENURE_CITATION = "POL-REMOTE-001 §3.1 Eligibility"
REMOTE_ROLE_CITATION = "POL-REMOTE-001 §3.4 Onsite-Required Roles"
PIP_CITATION = "POL-PERF-001 §8.3 Performance Improvement Plans"

# POL-PTO-001 §4 defines three separate blackout rules, and they only bite for
# requests of three or more consecutive days. They are encoded rather than
# hard-coded as date literals because the fiscal-close windows are *computed*
# (last 5 and first 3 business days around each quarter boundary) and would
# otherwise silently rot the moment the as-of date moves into another year.
BLACKOUT_MIN_CONSECUTIVE_DAYS = 3
FISCAL_QUARTER_ENDS = ((3, 31), (6, 30), (9, 30), (12, 31))
FISCAL_CLOSE_TRAILING_BUSINESS_DAYS = 5
FISCAL_CLOSE_LEADING_BUSINESS_DAYS = 3
CONFERENCE_WINDOWS = (("2026-09-14", "2026-09-18", "Helios Forum customer conference"),)
DEPARTMENT_BLACKOUTS = {
    "Customer Support": (("2026-11-15", "2027-01-05", "Peak support season"),),
}

VISA_WORK_AUTHORIZATIONS = {"visa_h1b", "visa_l1", "visa_tn"}
HOURS_PER_DAY = 8.0

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_ticket_counter = itertools.count(9001)
_created_tickets: list[dict] = []
_draft_emails: list[dict] = []


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load(name: str, key: str) -> list[dict]:
    if name not in _cache:
        with _lock:
            if name not in _cache:
                path = settings.mock_data_dir / f"{name}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                _cache[name] = payload[key]
                _cache[f"{name}:_meta"] = payload.get("_meta", {})
    return _cache[name]


def employees() -> list[dict]:
    return _load("employees", "employees")


def pto_balances() -> list[dict]:
    return _load("pto_balances", "balances")


def benefits_elections() -> list[dict]:
    return _load("benefits_elections", "elections")


def international_history() -> list[dict]:
    return _load("international_work_history", "records")


def offices() -> list[dict]:
    return _load("offices", "offices")


def hr_tickets() -> list[dict]:
    """Seed tickets plus any created this process (in-memory, never persisted)."""
    return _load("hr_tickets", "tickets") + _created_tickets


def today() -> dt.date:
    return dt.date.fromisoformat(settings.as_of_date)


def _parse_date(value: str, label: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def resolve_employee(identifier: str) -> dict | None:
    """Find an employee by id, email, full name, or unambiguous partial name.

    Tools are called with whatever the user typed ("Maya", "E-1041"), so
    resolution is forgiving -- but an ambiguous partial name returns a
    disambiguation error rather than a guess.
    """
    query = (identifier or "").strip()
    if not query:
        return None
    lowered = query.lower()

    for employee in employees():
        if employee["employee_id"].lower() == lowered or employee["email"].lower() == lowered:
            return employee
    for employee in employees():
        if employee["full_name"].lower() == lowered:
            return employee

    partial = [e for e in employees() if lowered in e["full_name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(f"{e['full_name']} ({e['employee_id']})" for e in partial)
        raise LookupError(f"{identifier!r} matches several employees: {names}")
    return None


def _row(rows: list[dict], employee_id: str) -> dict | None:
    return next((r for r in rows if r["employee_id"] == employee_id), None)


def employee_profile(identifier: str) -> dict:
    employee = resolve_employee(identifier)
    if employee is None:
        raise LookupError(f"no employee matches {identifier!r}")

    office = next((o for o in offices() if o["office_id"] == employee["office_location"]), None)
    manager = next((e for e in employees() if e["employee_id"] == employee.get("manager_id")), None)
    profile = dict(employee)
    profile["tenure_months"] = months_between(_parse_date(employee["hire_date"], "hire_date"), today())
    profile["office"] = office
    profile["manager_name"] = manager["full_name"] if manager else None
    profile["as_of_date"] = settings.as_of_date
    return profile


def months_between(start: dt.date, end: dt.date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def business_days_between(start: dt.date, end: dt.date) -> int:
    """Weekdays strictly after ``start`` up to and including ``end``.

    Company holidays are deliberately *not* subtracted: the policy says
    "business days" without excluding holidays, and inventing a stricter rule
    than the corpus states would make the agent's answer uncitable.
    """
    if end <= start:
        return 0
    days = 0
    cursor = start + dt.timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor += dt.timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# PTO
# ---------------------------------------------------------------------------
def pto_balance(identifier: str) -> dict:
    employee = employee_profile(identifier)
    balance = _row(pto_balances(), employee["employee_id"])
    if balance is None:
        raise LookupError(f"no PTO record for {employee['employee_id']}")

    result = dict(balance)
    result["employee_name"] = employee["full_name"]
    result["employment_type"] = employee["employment_type"]
    result["available_days"] = round(balance["available_hours"] / HOURS_PER_DAY, 2)
    result["as_of_date"] = settings.as_of_date
    if not balance["eligible"]:
        result["note"] = (
            f"{employee['full_name']} is {employee['employment_type']} and does not "
            f"accrue PTO (POL-BEN-001 §3 Eligibility by Employment Type)."
        )
    return result


def required_notice_days(consecutive_days: int) -> tuple[int, str]:
    for max_days, notice in PTO_NOTICE_BANDS:
        if consecutive_days <= max_days:
            band = (
                f"{consecutive_days} consecutive business day"
                f"{'s' if consecutive_days != 1 else ''}"
            )
            return notice, f"{band} requires {notice} business days notice ({PTO_NOTICE_CITATION})"
    raise AssertionError("unreachable: notice bands are open-ended")


def _shift_business_days(anchor: dt.date, count: int, backwards: bool) -> dt.date:
    """Walk ``count`` business days from ``anchor`` inclusive of ``anchor`` itself
    when it is a weekday."""
    step = dt.timedelta(days=-1 if backwards else 1)
    cursor = anchor
    remaining = count
    while True:
        if cursor.weekday() < 5:
            remaining -= 1
            if remaining <= 0:
                return cursor
        cursor += step


def fiscal_close_windows(year: int) -> list[tuple[dt.date, dt.date, str]]:
    """Blackout windows around each fiscal quarter boundary in ``year``."""
    windows: list[tuple[dt.date, dt.date, str]] = []
    for month, day in FISCAL_QUARTER_ENDS:
        quarter_end = dt.date(year, month, day)
        while quarter_end.weekday() >= 5:  # settle onto the last business day
            quarter_end -= dt.timedelta(days=1)
        start = _shift_business_days(quarter_end, FISCAL_CLOSE_TRAILING_BUSINESS_DAYS, backwards=True)
        end = _shift_business_days(
            quarter_end + dt.timedelta(days=1), FISCAL_CLOSE_LEADING_BUSINESS_DAYS, backwards=False
        )
        windows.append((start, end, f"Fiscal close for the quarter ending {quarter_end.isoformat()}"))
    return windows


def blackout_conflicts(department: str, start: dt.date, end: dt.date, days: float) -> list[dict]:
    """All POL-PTO-001 §4 blackout windows the requested dates overlap.

    Returns an empty list for requests shorter than three consecutive days: the
    policy restricts only absences of three or more days, and reporting a
    conflict for a one-day request would be a stricter rule than the corpus
    supports.
    """
    if days < BLACKOUT_MIN_CONSECUTIVE_DAYS:
        return []

    candidates: list[tuple[dt.date, dt.date, str]] = []
    for year in range(start.year, end.year + 1):
        candidates.extend(fiscal_close_windows(year))
    candidates.extend(
        (_parse_date(a, "conference start"), _parse_date(b, "conference end"), label)
        for a, b, label in CONFERENCE_WINDOWS
    )
    candidates.extend(
        (_parse_date(a, "blackout start"), _parse_date(b, "blackout end"), label)
        for a, b, label in DEPARTMENT_BLACKOUTS.get(department, ())
    )

    return [
        {
            "window": label,
            "blackout_start": w_start.isoformat(),
            "blackout_end": w_end.isoformat(),
            "citation": PTO_BLACKOUT_CITATION,
        }
        for w_start, w_end, label in candidates
        if start <= w_end and end >= w_start
    ]


def check_pto_request(identifier: str, start_date: str, days: float) -> dict:
    """Evaluate a PTO request against balance, notice, and blackout rules."""
    employee = employee_profile(identifier)
    start = _parse_date(start_date, "start_date")
    days = float(days)
    if days <= 0:
        raise ValueError("days must be greater than zero")

    hours_requested = days * HOURS_PER_DAY
    end = start + dt.timedelta(days=max(int(round(days)) - 1, 0))
    balance = pto_balance(employee["employee_id"])

    findings: list[dict] = []

    if not balance["eligible"]:
        findings.append({
            "check": "eligibility",
            "status": "fail",
            "detail": balance.get("note", "not eligible for PTO"),
            "citation": "POL-BEN-001 §3 Eligibility by Employment Type",
        })

    available = float(balance.get("available_hours", 0.0))
    sufficient = available >= hours_requested
    findings.append({
        "check": "balance",
        "status": "pass" if sufficient else "fail",
        "detail": (
            f"request is {hours_requested:g} hours ({days:g} days); "
            f"{available:g} hours ({balance['available_days']:g} days) available "
            f"after {balance.get('scheduled_future_hours', 0):g} hours already scheduled"
        ),
        "shortfall_hours": round(max(hours_requested - available, 0.0), 2),
        "citation": PTO_BALANCE_CITATION,
    })

    notice_required, notice_detail = required_notice_days(int(round(days)))
    notice_given = business_days_between(today(), start)
    findings.append({
        "check": "notice",
        "status": "pass" if notice_given >= notice_required else "fail",
        "detail": (
            f"{notice_given} business days notice given (today {settings.as_of_date} "
            f"-> {start_date}); {notice_detail}"
        ),
        "required_business_days": notice_required,
        "given_business_days": notice_given,
        "citation": PTO_NOTICE_CITATION,
    })

    conflicts = blackout_conflicts(employee["department"], start, end, days)
    findings.append({
        "check": "blackout",
        "status": "fail" if conflicts else "pass",
        "detail": (
            "; ".join(
                f"{c['window']} ({c['blackout_start']} to {c['blackout_end']}) "
                f"overlaps the requested dates"
                for c in conflicts
            )
            if conflicts else
            (
                f"no blackout period applies to {employee['department']} for these dates"
                if days >= BLACKOUT_MIN_CONSECUTIVE_DAYS else
                f"blackout periods restrict absences of {BLACKOUT_MIN_CONSECUTIVE_DAYS} or more "
                f"consecutive days; this request is {days:g}"
            )
        ),
        "conflicts": conflicts,
        "exception_path": (
            "VP-level approval is required to take PTO during a blackout period"
            if conflicts else None
        ),
        "citation": PTO_BLACKOUT_CITATION,
    })

    failures = [f for f in findings if f["status"] == "fail"]
    return {
        "employee_id": employee["employee_id"],
        "employee_name": employee["full_name"],
        "request": {
            "start_date": start_date,
            "end_date": end.isoformat(),
            "days": days,
            "hours": hours_requested,
        },
        "as_of_date": settings.as_of_date,
        "compliant": not failures,
        "findings": findings,
        "blocking_reasons": [f["detail"] for f in failures],
        "citations": sorted({f["citation"] for f in findings}),
    }


# ---------------------------------------------------------------------------
# International remote work
# ---------------------------------------------------------------------------
def international_usage(identifier: str, as_of: str | None = None) -> dict:
    """Days consumed inside the rolling 12-month window.

    The rolling window is the single easiest rule in this system to get wrong:
    summing every historical record double-counts trips that have aged out. The
    excluded records are returned explicitly so the agent -- and the grader --
    can see which trips were discounted and why.
    """
    employee = employee_profile(identifier)
    reference = _parse_date(as_of, "as_of") if as_of else today()
    window_start = reference - dt.timedelta(days=INTL_ROLLING_WINDOW_DAYS)

    counted, excluded = [], []
    for record in international_history():
        if record["employee_id"] != employee["employee_id"]:
            continue
        if record["status"] not in ("approved", "completed"):
            excluded.append({**record, "excluded_because": f"status is {record['status']}"})
            continue
        end = _parse_date(record["end_date"], "end_date")
        if end < window_start:
            excluded.append({
                **record,
                "excluded_because": (
                    f"ended {record['end_date']}, before the rolling window opened "
                    f"on {window_start.isoformat()}"
                ),
            })
            continue
        counted.append(record)

    used = sum(r["calendar_days"] for r in counted)
    return {
        "employee_id": employee["employee_id"],
        "employee_name": employee["full_name"],
        "work_authorization": employee["work_authorization"],
        "as_of_date": reference.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": reference.isoformat(),
        "annual_limit_days": INTL_ANNUAL_LIMIT_DAYS,
        "days_used": used,
        "days_remaining": INTL_ANNUAL_LIMIT_DAYS - used,
        "counted_records": counted,
        "excluded_records": excluded,
        "citation": INTL_LIMIT_CITATION,
    }


def check_international_request(
    identifier: str, country: str, start_date: str, end_date: str
) -> dict:
    employee = employee_profile(identifier)
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date must not be before start_date")

    requested_days = (end - start).days + 1
    usage = international_usage(employee["employee_id"], as_of=start_date)
    projected = usage["days_used"] + requested_days

    findings: list[dict] = [{
        "check": "rolling_limit",
        "status": "pass" if projected <= INTL_ANNUAL_LIMIT_DAYS else "fail",
        "detail": (
            f"{requested_days} requested days plus {usage['days_used']} already used in the "
            f"window {usage['window_start']} to {usage['window_end']} = {projected} days "
            f"against a {INTL_ANNUAL_LIMIT_DAYS}-day limit"
        ),
        "requested_days": requested_days,
        "days_already_used": usage["days_used"],
        "projected_total_days": projected,
        "overage_days": max(projected - INTL_ANNUAL_LIMIT_DAYS, 0),
        "citation": INTL_LIMIT_CITATION,
    }]

    needs_immigration = employee["work_authorization"] in VISA_WORK_AUTHORIZATIONS
    findings.append({
        "check": "immigration",
        "status": "review_required" if needs_immigration else "pass",
        "detail": (
            f"work authorization is {employee['work_authorization']}; Immigration Counsel "
            f"must review before any international work is approved"
            if needs_immigration else
            f"work authorization is {employee['work_authorization']}; no immigration review triggered"
        ),
        "citation": INTL_IMMIGRATION_CITATION,
    })

    tenure_ok = employee["tenure_months"] >= REMOTE_TENURE_MONTHS
    findings.append({
        "check": "tenure",
        "status": "pass" if tenure_ok else "fail",
        "detail": (
            f"{employee['tenure_months']} months of service since {employee['hire_date']}; "
            f"{REMOTE_TENURE_MONTHS} months required"
        ),
        "citation": REMOTE_TENURE_CITATION,
    })

    blocking = [f for f in findings if f["status"] == "fail"]
    review = [f for f in findings if f["status"] == "review_required"]
    return {
        "employee_id": employee["employee_id"],
        "employee_name": employee["full_name"],
        "request": {
            "country": country,
            "start_date": start_date,
            "end_date": end_date,
            "calendar_days": requested_days,
        },
        "usage": {k: usage[k] for k in
                  ("window_start", "window_end", "days_used", "days_remaining",
                   "counted_records", "excluded_records")},
        "compliant": not blocking and not review,
        "requires_review": bool(review),
        "findings": findings,
        "blocking_reasons": [f["detail"] for f in blocking],
        "citations": sorted({f["citation"] for f in findings}),
    }


# ---------------------------------------------------------------------------
# Remote-work arrangement changes
# ---------------------------------------------------------------------------
def check_remote_arrangement_change(identifier: str, requested_arrangement: str) -> dict:
    employee = employee_profile(identifier)
    requested = (requested_arrangement or "").strip().lower().replace("-", "_").replace(" ", "_")
    if requested not in ("onsite", "hybrid", "remote"):
        raise ValueError("requested_arrangement must be one of: onsite, hybrid, remote")

    findings: list[dict] = []

    tenure_ok = employee["tenure_months"] >= REMOTE_TENURE_MONTHS
    findings.append({
        "check": "tenure",
        "status": "pass" if tenure_ok else "fail",
        "detail": (
            f"{employee['tenure_months']} months of service since {employee['hire_date']}; "
            f"{REMOTE_TENURE_MONTHS} months required for a remote or hybrid arrangement"
        ),
        "citation": REMOTE_TENURE_CITATION,
    })

    role_ok = employee.get("remote_eligible_role", True)
    findings.append({
        "check": "role_eligibility",
        "status": "pass" if role_ok else "fail",
        "detail": (
            f"{employee['title']} is designated remote-eligible"
            if role_ok else
            f"{employee['title']} is designated Onsite Required in the job architecture"
        ),
        "citation": REMOTE_ROLE_CITATION,
    })

    on_pip = bool(employee.get("on_pip"))
    findings.append({
        "check": "performance",
        "status": "fail" if on_pip else "pass",
        "detail": (
            "an active Performance Improvement Plan blocks a change to a fully remote "
            "arrangement until the plan closes successfully"
            if on_pip else
            f"latest rating is {employee.get('latest_performance_rating') or 'not yet reviewed'}; "
            "no active PIP"
        ),
        "citation": PIP_CITATION,
    })

    applicable = findings if requested == "remote" else [
        f for f in findings if f["check"] != "performance"
    ]
    blocking = [f for f in applicable if f["status"] == "fail"]
    return {
        "employee_id": employee["employee_id"],
        "employee_name": employee["full_name"],
        "current_arrangement": employee["work_arrangement"],
        "requested_arrangement": requested,
        "compliant": not blocking,
        "findings": applicable,
        "blocking_reasons": [f["detail"] for f in blocking],
        "citations": sorted({f["citation"] for f in applicable}),
    }


# ---------------------------------------------------------------------------
# Benefits
# ---------------------------------------------------------------------------
def benefits_status(identifier: str) -> dict:
    employee = employee_profile(identifier)
    election = _row(benefits_elections(), employee["employee_id"])
    if election is None:
        raise LookupError(f"no benefits record for {employee['employee_id']}")

    result = dict(election)
    result["employee_name"] = employee["full_name"]
    result["employment_type"] = employee["employment_type"]
    result["weekly_hours"] = employee["weekly_hours"]
    result["as_of_date"] = settings.as_of_date
    if not election["benefits_eligible"]:
        result["note"] = (
            f"{employee['full_name']} is {employee['employment_type']} and is not "
            f"benefits-eligible (POL-BEN-001 §3 Eligibility by Employment Type)."
        )
    return result


# ---------------------------------------------------------------------------
# Mock writes -- in-memory only
# ---------------------------------------------------------------------------
def create_ticket(
    identifier: str, category: str, subject: str, body: str, priority: str = "normal"
) -> dict:
    employee = employee_profile(identifier)
    if not subject.strip():
        raise ValueError("subject must not be empty")

    ticket = {
        "ticket_id": f"HR-2026-{next(_ticket_counter)}",
        "employee_id": employee["employee_id"],
        "employee_name": employee["full_name"],
        "category": category,
        "subject": subject.strip(),
        "body": body.strip(),
        "status": "open",
        "priority": priority if priority in ("low", "normal", "high", "urgent") else "normal",
        "assigned_team": _route(category),
        "created_at": f"{settings.as_of_date}T00:00:00Z",
        "resolved_at": None,
        "persisted": False,
        "note": "MOCK ticket. Held in memory for this process only; hr_tickets.json is untouched.",
    }
    _created_tickets.append(ticket)
    return ticket


def _route(category: str) -> str:
    return {
        "international_remote_work": "Global Mobility",
        "remote_work": "HR Business Partner",
        "pto": "HR Operations",
        "benefits": "Benefits Team",
        "equipment": "IT Asset Management",
        "expense": "Finance",
        "security": "Security Operations",
    }.get(category, "HR Operations")


def draft_email(to_identifier: str, subject: str, body: str) -> dict:
    employee = employee_profile(to_identifier)
    draft = {
        "draft_id": f"DRAFT-{len(_draft_emails) + 1:04d}",
        "to": employee["email"],
        "to_name": employee["full_name"],
        "subject": subject.strip(),
        "body": body.strip(),
        "created_at": f"{settings.as_of_date}T00:00:00Z",
        "sent": False,
        "note": "MOCK draft. Nothing is sent; no mail transport is configured or reachable.",
    }
    _draft_emails.append(draft)
    return draft


def reset_writes() -> None:
    """Clear in-memory writes. Used by tests to keep cases independent."""
    _created_tickets.clear()
    _draft_emails.clear()
