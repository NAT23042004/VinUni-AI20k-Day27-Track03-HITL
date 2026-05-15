"""Helpers for scripted or interactive HITL resumes."""

from __future__ import annotations

from typing import Callable


def scripted_approval_response(choice: str | None, feedback: str) -> dict | None:
    if choice is None:
        return None
    return {"choice": choice, "feedback": feedback}


def scripted_escalation_response(
    questions: list[str],
    answers: list[str],
    default_answer: str | None,
) -> dict | None:
    if not answers and default_answer is None:
        return None

    response: dict[str, str] = {}
    for index, question in enumerate(questions):
        if index < len(answers):
            response[question] = answers[index]
        elif default_answer is not None:
            response[question] = default_answer
        else:
            raise RuntimeError(
                "Not enough scripted escalation answers were provided for this interrupt."
            )
    return response


def interactive_or_raise(payload: dict, prompt: Callable[[dict], dict]) -> dict:
    try:
        return prompt(payload)
    except EOFError as exc:
        raise RuntimeError(
            "Interactive input is unavailable. Re-run in a real terminal or provide "
            "scripted resume flags for this exercise."
        ) from exc
