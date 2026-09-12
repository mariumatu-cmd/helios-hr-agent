"""MCP server exposing Helios HR policy and operations tools.

Run it directly for stdio (what the agent uses locally and in CI):

    python mcp_server/hr_server.py

or over Streamable HTTP, which is what the deployed service uses so that the
agent and the server are genuinely separate processes speaking MCP rather than
in-process function calls:

    MCP_TRANSPORT=streamable-http python mcp_server/hr_server.py

Design notes
------------
*Thin tools, thick rules.* Every tool is a JSON-serialising wrapper around
``mcp_server.data`` or ``rag.retrieve``. No policy arithmetic happens here, so
the same logic is exercised by the unit tests without an MCP round trip.

*Errors are values.* A bad employee id returns a structured ``{"error": ...}``
payload with a hint, not an exception. The agent can then correct itself on the
next step instead of the whole run failing -- this is the recovery path
demonstrated in the evaluation suite.

*Writes are gated and reversible.* ``create_hr_ticket`` and ``draft_hr_email``
refuse to act unless called with ``confirmed=true``; unconfirmed calls return a
preview of exactly what would happen. Even when confirmed, they only mutate
in-memory state. Both are advertised with MCP ``ToolAnnotations`` so a client
can see they are non-read-only *before* calling them.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Callable
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from config import settings  # noqa: E402
from mcp_server import data  # noqa: E402
from rag import retrieve  # noqa: E402

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
MOCK_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,   # in-memory only; nothing on disk is modified
    idempotentHint=False,
    openWorldHint=False,
)

mcp = MCPServer(
    name="helios-hr",
    version="1.0.0",
    instructions=(
        "Helios Systems HR assistant tools. Twelve HR policy documents are searchable "
        "with `search_policy_documents`; employee-specific facts come from the mock HR "
        "system via the lookup tools. Always ground a policy claim in a retrieved "
        "passage and quote its `citation`. Use `check_policy_compliance` rather than "
        "doing date or balance arithmetic yourself -- it is deterministic and returns "
        "the citation for every rule it applies. Ticket creation and email drafting are "
        "mock actions and require an explicit `confirmed=true`."
    ),
)


def _ok(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(message: str, hint: str = "") -> str:
    return json.dumps({"error": message, "hint": hint}, ensure_ascii=False)


def _guard(fn: Callable[[], Any]) -> str:
    """Turn expected failures into structured results the agent can act on."""
    try:
        return _ok(fn())
    except LookupError as exc:
        return _error(str(exc), "Call list_employees to see valid identifiers.")
    except ValueError as exc:
        return _error(str(exc), "Check the argument format and try again.")
    except FileNotFoundError as exc:
        return _error(str(exc), "The RAG index is missing; run python -m rag.ingest.build_index.")


# ---------------------------------------------------------------------------
# Policy retrieval (RAG)
# ---------------------------------------------------------------------------
@mcp.tool(annotations=READ_ONLY)
def search_policy_documents(query: str, k: int = 5, doc_id: str | None = None) -> str:
    """Search the Helios HR policy corpus and return grounded passages with citations.

    This is the first tool to call for any question about what a policy says.

    Args:
        query: a natural-language question, e.g. "how much notice for three days off".
        k: number of passages to return (1-10).
        doc_id: optional filter, e.g. "POL-PTO-001", to search within one document.

    Returns JSON with `grounded` (false means the corpus does not cover the
    question and you must say so rather than answer), `best_similarity`, and a
    list of `hits`, each carrying the exact `citation` string to quote.
    """
    def run() -> Any:
        result = retrieve.search(query, k=max(1, min(int(k), 10)), doc_id=doc_id)
        payload = result.to_dict()
        if not payload["grounded"]:
            payload["instruction"] = (
                "Do not answer from these passages. Tell the user the HR policy corpus "
                "does not cover this question and suggest contacting HR directly."
            )
        return payload

    return _guard(run)


@mcp.tool(annotations=READ_ONLY)
def get_policy_section(doc_id: str, section: str) -> str:
    """Fetch a specific policy section verbatim, by number or heading text.

    Use after `search_policy_documents` when a passage references another
    section (e.g. "see Section 4.4") and you need that section in full.

    Args:
        doc_id: document identifier, e.g. "POL-INTL-001".
        section: section number such as "4.4", or part of the heading such as "Immigration".
    """
    def run() -> Any:
        hits = retrieve.get_section(doc_id, section)
        if not hits:
            return {
                "error": f"no section matching {section!r} in {doc_id}",
                "hint": "Call list_policy_documents to see the sections in this document.",
            }
        return {"doc_id": doc_id, "section": section, "passages": [h.to_dict() for h in hits]}

    return _guard(run)


@mcp.tool(annotations=READ_ONLY)
def list_policy_documents() -> str:
    """List every indexed HR policy document with its sections, version and effective date.

    Useful for orienting before a search, or for telling a user what the
    assistant is able to answer questions about.
    """
    return _guard(lambda: {
        "index": retrieve.index_info(),
        "documents": retrieve.list_documents(),
    })


# ---------------------------------------------------------------------------
# Employee records (mock structured data)
# ---------------------------------------------------------------------------
@mcp.tool(annotations=READ_ONLY)
def lookup_employee_profile(employee: str) -> str:
    """Look up one employee's record: role, tenure, location, work arrangement, visa status.

    Args:
        employee: employee id ("E-1041"), email, or full/partial name ("Maya Rodriguez").

    All data is synthetic. Returns `tenure_months`, `work_authorization`,
    `on_pip` and office details -- the facts most policy rules depend on.
    """
    return _guard(lambda: data.employee_profile(employee))


@mcp.tool(annotations=READ_ONLY)
def list_employees(department: str | None = None) -> str:
    """List employees in the mock HR system, optionally filtered by department.

    Args:
        department: optional exact department name, e.g. "Engineering".
    """
    def run() -> Any:
        people = data.employees()
        if department:
            wanted = department.strip().lower()
            people = [e for e in people if e["department"].lower() == wanted]
        return {
            "count": len(people),
            "employees": [
                {k: e[k] for k in
                 ("employee_id", "full_name", "title", "department",
                  "employment_type", "work_arrangement", "office_location")}
                for e in people
            ],
        }

    return _guard(run)


@mcp.tool(annotations=READ_ONLY)
def check_pto_balance(employee: str) -> str:
    """Get an employee's current PTO balance, accrual rate and available hours.

    Args:
        employee: employee id, email, or name.

    `available_hours` (balance minus already-scheduled future time off) is the
    figure a new request draws from -- use it, not `balance_hours`.
    """
    return _guard(lambda: data.pto_balance(employee))


@mcp.tool(annotations=READ_ONLY)
def lookup_benefits_status(employee: str) -> str:
    """Get an employee's benefits eligibility and current elections.

    Args:
        employee: employee id, email, or name.

    Covers medical/dental/vision, HSA and FSA elections, retirement deferral,
    disability cover and parental-leave eligibility.
    """
    return _guard(lambda: data.benefits_status(employee))


@mcp.tool(annotations=READ_ONLY)
def check_international_work_usage(employee: str, as_of: str | None = None) -> str:
    """Days an employee has used against the 30-day rolling international-work limit.

    Args:
        employee: employee id, email, or name.
        as_of: optional ISO date the rolling 12-month window is measured back from.

    Returns both `counted_records` and `excluded_records` with the reason each
    trip was excluded, so the total can be explained rather than asserted.
    Trips that ended before the window opened do not count.
    """
    return _guard(lambda: data.international_usage(employee, as_of=as_of))


# ---------------------------------------------------------------------------
# Composite rule evaluation
# ---------------------------------------------------------------------------
@mcp.tool(annotations=READ_ONLY)
def check_policy_compliance(
    request_type: str,
    employee: str,
    start_date: str | None = None,
    end_date: str | None = None,
    days: float | None = None,
    country: str | None = None,
    arrangement: str | None = None,
) -> str:
    """Deterministically evaluate a request against every applicable policy rule.

    Prefer this over reasoning about dates or balances yourself: it applies the
    rules exactly, and returns a `citation` for each one so the answer can be
    sourced.

    Args:
        request_type: "pto", "international_remote_work", or "remote_arrangement".
        employee: employee id, email, or name.
        start_date: ISO date. Required for "pto" and "international_remote_work".
        end_date: ISO date. Required for "international_remote_work".
        days: number of PTO days. Required for "pto".
        country: destination country. Required for "international_remote_work".
        arrangement: "onsite", "hybrid" or "remote". Required for "remote_arrangement".

    Returns `compliant`, `findings` (one per rule, each pass/fail/review_required),
    `blocking_reasons` and `citations`.
    """
    def run() -> Any:
        kind = (request_type or "").strip().lower()
        if kind == "pto":
            if not start_date or days is None:
                raise ValueError("pto requests require start_date and days")
            return data.check_pto_request(employee, start_date, days)
        if kind in ("international_remote_work", "international"):
            if not (start_date and end_date and country):
                raise ValueError(
                    "international_remote_work requires country, start_date and end_date"
                )
            return data.check_international_request(employee, country, start_date, end_date)
        if kind in ("remote_arrangement", "remote_work"):
            if not arrangement:
                raise ValueError("remote_arrangement requires arrangement")
            return data.check_remote_arrangement_change(employee, arrangement)
        raise ValueError(
            "request_type must be one of: pto, international_remote_work, remote_arrangement"
        )

    return _guard(run)


# ---------------------------------------------------------------------------
# Mock write actions -- confirmation gated
# ---------------------------------------------------------------------------
@mcp.tool(annotations=MOCK_WRITE)
def create_hr_ticket(
    employee: str,
    category: str,
    subject: str,
    body: str,
    priority: str = "normal",
    confirmed: bool = False,
) -> str:
    """Create a MOCK HR ticket. Requires confirmed=true; otherwise returns a preview.

    Call once with `confirmed=false` (the default) to show the user exactly what
    would be filed and which team it routes to, then call again with
    `confirmed=true` only after the user agrees.

    Args:
        employee: employee id, email, or name the ticket is filed for.
        category: one of international_remote_work, remote_work, pto, benefits,
            equipment, expense, security.
        subject: one-line summary.
        body: the ticket detail, including the policy citations that justify it.
        priority: low, normal, high or urgent.
        confirmed: must be true to actually create the ticket.

    Nothing is persisted: the ticket exists in memory for this process only.
    """
    def run() -> Any:
        profile = data.employee_profile(employee)
        if not confirmed:
            return {
                "requires_confirmation": True,
                "action": "create_hr_ticket",
                "preview": {
                    "employee_id": profile["employee_id"],
                    "employee_name": profile["full_name"],
                    "category": category,
                    "subject": subject,
                    "body": body,
                    "priority": priority,
                    "would_route_to": data._route(category),
                },
                "instruction": (
                    "Show this preview to the user and ask them to confirm. Call again "
                    "with confirmed=true only if they agree."
                ),
            }
        return data.create_ticket(employee, category, subject, body, priority)

    return _guard(run)


@mcp.tool(annotations=MOCK_WRITE)
def draft_hr_email(employee: str, subject: str, body: str, confirmed: bool = False) -> str:
    """Draft a MOCK email to an employee. Requires confirmed=true; otherwise returns a preview.

    Args:
        employee: recipient employee id, email, or name.
        subject: email subject line.
        body: email body. Include the policy citations that support it.
        confirmed: must be true to store the draft.

    No mail is ever sent and no transport is configured; the draft is held in
    memory only.
    """
    def run() -> Any:
        profile = data.employee_profile(employee)
        if not confirmed:
            return {
                "requires_confirmation": True,
                "action": "draft_hr_email",
                "preview": {"to": profile["email"], "subject": subject, "body": body},
                "instruction": (
                    "Show this draft to the user and ask them to confirm before saving it."
                ),
            }
        return data.draft_email(employee, subject, body)

    return _guard(run)


@mcp.tool(annotations=READ_ONLY)
def list_hr_tickets(employee: str | None = None) -> str:
    """List HR tickets: the seeded history plus any created during this session.

    Args:
        employee: optional employee id, email, or name to filter by.
    """
    def run() -> Any:
        tickets = data.hr_tickets()
        if employee:
            profile = data.employee_profile(employee)
            tickets = [t for t in tickets if t["employee_id"] == profile["employee_id"]]
        return {"count": len(tickets), "tickets": tickets}

    return _guard(run)


if __name__ == "__main__":
    transport = settings.mcp_transport.strip().lower()
    if transport not in ("stdio", "sse", "streamable-http"):
        raise SystemExit(
            f"MCP_TRANSPORT={transport!r} is not supported; "
            f"use stdio, sse or streamable-http"
        )
    mcp.run(transport=transport)
