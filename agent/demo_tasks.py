"""Canonical deployed demo tasks, shared by the UI and verifier."""

DEMO_WORKFLOWS = [
    {
        "id": "international",
        "label": "1. International remote work",
        "question": (
            "Maya Rodriguez wants to work from Portugal for 42 calendar days, "
            "5 October to 15 November 2026. Can she? Check her profile and rolling "
            "international usage, retrieve the international and remote-work policies, "
            "and explain the limit and any compliant options with citations."
        ),
    },
    {
        "id": "pto",
        "label": "2. PTO guidance and ticket preview",
        "question": (
            "Jonas Weber wants three business days of PTO starting 21 September 2026. "
            "Retrieve the PTO policy, check his available balance and compliance, "
            "and explain both the balance and notice issues with citations. "
            "Prepare a mock HR ticket preview asking about the permitted alternatives; "
            "do not create it until I use the confirmation button."
        ),
    },
]
