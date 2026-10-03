"""Canonical deployed demo tasks, shared by the UI and verifier.

Each prompt names the checks and the order it expects. The free tier caps each
model's tokens per minute, so a multi-step task usually finishes on a smaller
fallback model, and an explicit sequence keeps it on the full workflow: the
compliance check, citations from every policy involved and, for PTO, the ticket
preview before the final explanation.
"""

DEMO_WORKFLOWS = [
    {
        "id": "international",
        "label": "1. International remote work",
        "question": (
            "Maya Rodriguez wants to work from Portugal for 42 calendar days, "
            "5 October to 15 November 2026. Can she? Check her rolling international "
            "usage, run the policy compliance check for this request, and retrieve both "
            "the international remote-work policy and the remote-work policy's rules for "
            "working outside the home country. Explain the limit and any compliant "
            "options, citing both policies."
        ),
    },
    {
        "id": "pto",
        "label": "2. PTO guidance and ticket preview",
        "question": (
            "Jonas Weber wants three business days of PTO starting 21 September 2026. "
            "Check his PTO balance and run the policy compliance check, then retrieve "
            "the PTO policy's notice and insufficient-balance sections. Prepare a mock "
            "HR ticket preview asking about the permitted alternatives, but do not "
            "create it until I use the confirmation button. Finally, explain both the "
            "balance and notice issues with citations."
        ),
    },
]
