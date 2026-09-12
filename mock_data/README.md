# Mock Data

All data in this directory is **synthetic**. Every person, email address,
identifier, balance, and ticket was fabricated for this academic project. No
real, private, or proprietary employee information is present.

Email addresses use the reserved `example.com` domain (RFC 2606) and phone
numbers use the reserved `555-01xx` range, so nothing here can resolve to a real
contact.

## Files

| File | Records | Purpose |
| --- | --- | --- |
| `employees.json` | 16 | Employee profiles, employment type, manager chain, location, work arrangement, performance state, work authorization |
| `pto_balances.json` | 16 | PTO accrual band, balance, scheduled future time, floating holidays |
| `benefits_elections.json` | 16 | Plan elections, waiting-period state, match and parental-leave eligibility |
| `international_work_history.json` | 5 | Approved international remote-work days, for the rolling 12-month limit |
| `offices.json` | 7 | Office locations, timezones, holiday schedule, expense tier |
| `hr_tickets.json` | 5 | Seed HR tickets, plus the ID sequence for mock ticket creation |

## Design notes

The data is deliberately shaped so that agentic workflows require real
computation against policy, not a single lookup:

- **`E-1041` Maya Rodriguez** has used **12** international days inside the
  current rolling 12-month window (Ireland, March 2026). Her Spain trip
  (`IRW-2025-0298`, August 2025) is **outside** the window and must not be
  counted — a naive implementation double-counts it and reports 24.
- **`E-1088` Jonas Weber** has only **13 PTO hours** available, so a 3-day
  request must be refused with alternatives (`POL-PTO-001` §3.3). He is also in
  Customer Support, which has a November 15 – January 5 blackout.
- **`E-1120` Tomas Silva** has under 6 months of service and is on an H-1B, so
  he fails both the fully-remote tenure test (`POL-REMOTE-001` §3.1) and
  requires Immigration Counsel review (`POL-INTL-001` §4.4).
- **`E-1055` Marcus Doyle** is on an active PIP, which blocks a change to fully
  remote (`POL-PERF-001` §8.3).
- **`E-1099` Sofia Marino** is part-time: no STD/LTD, pro-rated parental leave.
- **`E-1150`**, **`E-1177`**, and **`C-2003`** are temporary, intern, and
  contractor respectively, and are benefit-ineligible.

## Mutability

`hr_tickets.json` is **seed state only**. Tickets created through the agent's
`create_mock_hr_ticket` tool are written to an in-memory store and are never
persisted back to disk. This keeps every agent action reversible and the
repository reproducible, satisfying the "prevent irreversible actions"
requirement.
