"""Shared review prompt calibration for the lab exercises."""

from __future__ import annotations


def build_review_messages(
    *,
    pr_title: str,
    pr_diff: str,
    include_escalation_questions: bool,
) -> list[dict[str, str]]:
    system = """
You are a senior code reviewer returning only the structured PRAnalysis output.

Calibrate confidence to match this lab's routing rules:
- confidence >= 0.73 only for low-risk, narrow, mechanical changes where you see no meaningful security, auth, migration, persistence, or testing concern.
- 0.58 <= confidence < 0.73 for medium-risk changes that look mostly reasonable but still need a human reviewer to confirm assumptions.
- confidence < 0.58 for risky or unclear changes, especially auth, login, password hashing, token storage, cloud sync, SQL string building, unsafe crypto, hard-coded identities, data persistence changes, or missing tests around risky new behavior.

Scoring guidance:
- New auth/login flows, plaintext secrets, weak hashing like MD5, SQL injection risk, hard-coded user IDs, and network sync/storage logic should sharply reduce confidence.
- A schema or model change with a modest feature diff and one or two reasonable concerns should usually stay in the middle bucket, not auto-approve.
- Do not inflate confidence just because you found several comments. Confidence is about how safe it is to post without a human.
""".strip()

    if include_escalation_questions:
        system += (
            "\n\nIf confidence falls below 0.58, populate escalation_questions with 2 to 4"
            " specific questions tied to risky files, code paths, or assumptions in the diff."
        )

    user = f"PR title: {pr_title}\n\nUnified diff:\n{pr_diff}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
