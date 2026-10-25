"""What the agent did with the tools, and whether it stayed inside the contract.

An orchestrator is usually described as the part with no natural test.  That
is true of its prose and false of its behaviour.  The three invariants below
are not statements about how good an answer reads; they are statements about
which tools were called, what each one returned, and what survived into the
output.  Every one of those is observable, so every one is testable --
against the record of the calls rather than against the text.

* **I1** A model may describe a number; it may never supply one.
* **I2** Severity propagates and never flattens: every issue a tool
  reported appears in the answer, under its own level.
* **I3** A refusal is terminal.  The same question is not asked again with
  the knobs moved.

This module keeps the record and turns each invariant into a function that
either returns an empty list or names what broke.  It computes nothing about
DNA, imports no other module in the package, and is deliberately ignorant of
what any particular tool means: it watches shapes, not biology.

The division of labour with `agent.py`: I3 is a gate, so it is enforced
there, at the moment of the offending call, and re-checkable here
afterwards.  I1 and I2 are properties of the finished answer, so they can
only be audited here, once there is an answer to audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass

# Every refusal in the system, by class name.  Held as names rather than
# classes so that this module stays free of the package it is watching.
TERMINAL = frozenset(
    {
        "DesignFailed",
        "SequenceError",
        "VectorError",
        "EnzymeError",
        "ThermoError",
    }
)

LEVELS = ("error", "warning", "note")

# How `report` marks each level, so an audit can read a rendered block back.
LEVEL_MARK = {"error": "FAIL", "warning": "WARN", "note": "note"}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class InvariantError(RuntimeError):
    """A tool call that the contract does not allow.

    Raised at the call, not collected at the end, because the invariants
    this covers (I3, I4) are about an action being taken rather than an
    answer being written.  By the time a refusal has been routed around
    there is nothing left to audit -- the damage is that the call happened.
    """

    def __init__(self, invariant: str, detail: str):
        self.invariant = invariant
        self.detail = detail
        super().__init__(f"{invariant}: {detail}")


# ---------------------------------------------------------------------------
# reading a result
# ---------------------------------------------------------------------------


def _walk(obj, seen: set[int] | None = None, depth: int = 0):
    """Yield every object reachable from `obj` by declared structure.

    Dataclass fields, `vars()`, and the elements of lists, tuples, sets and
    dicts -- and nothing else.  Properties are skipped on purpose: `Design`
    computes `construct`, `amplicon` and `backbone` on access, and a walker
    that touched them would do real sequence work just to take an inventory.

    `seen` holds `id()`s, and that is sound here for a reason worth
    writing down, because the same assumption was unsound elsewhere in
    this package and had to be fixed.  An `id()` is unique only among
    objects that are alive, so a set of them is only meaningful while
    everything in it is still alive.  Here everything is: this walks
    stored attributes and container elements, never computed ones, so
    every object reached is held by its parent and the whole chain is
    held by the caller's root for the length of the traversal.  Nothing
    walked can be collected while the walk is running, so no id can be
    reused inside it, and `seen` is fresh per call.  If this ever starts
    following properties, that argument collapses with it.
    """
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj

    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(key, seen, depth + 1)
            yield from _walk(value, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            yield from _walk(item, seen, depth + 1)
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            yield from _walk(getattr(obj, f.name, None), seen, depth + 1)
        return
    for value in vars(obj).values() if hasattr(obj, "__dict__") else ():
        yield from _walk(value, seen, depth + 1)


def issues_in(obj) -> tuple:
    """Every `Issue` reachable from a result, at whatever depth it sits.

    Deliberately duck-typed and deliberately deep.  Issues do not live in
    one place: a `Design` carries its own, a `Protocol` carries some on the
    plan and some on individual steps, a `ColonyPlan` carries some per
    assay.  I2 protects all of them equally -- an issue buried on step four
    is as much a part of the answer as one at the top -- so the inventory
    has to go all the way down rather than reading a single `.issues`.
    """
    found = []
    for item in _walk(obj):
        level = getattr(item, "level", None)
        text = getattr(item, "text", None)
        if isinstance(level, str) and isinstance(text, str) and level in LEVELS:
            found.append(item)
    return tuple(found)


def numbers_in_text(text: str) -> set[str]:
    """Every numeric token in a string, as written."""
    return set(_NUMBER.findall(text))


def numbers_in(obj) -> set[float]:
    """Every number a result carries, from its fields and from its strings.

    Strings count because Layer 1 puts numbers into prose as well as into
    floats: an issue text says how many bases were lost, a vector is named
    `pET-28a(+)`, an accession is `NM_001183088`.  Those digits originated
    inside the toolchain just as much as `Design.core_start` did, and I1 is
    about origin, not about datatype.
    """
    pool: set[float] = set()
    for item in _walk(obj):
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            pool.add(float(item))
        elif isinstance(item, str):
            for token in _NUMBER.findall(item):
                pool.add(float(token))
    return pool


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One tool call, as it happened."""

    tool: str
    layer: int
    # What was asked, minus the arguments a retry would mutate.  Two calls
    # with the same target are asking the same question.
    target: tuple
    # The arguments that *are* tunable -- `Options`, mostly.  Kept apart so
    # that I3 can tell a different question from the same question again.
    tuned: tuple
    outcome: str  # "returned" | "declined"
    declined_with: str = ""
    issues: tuple = ()
    numbers: frozenset = frozenset()
    rendered: str = ""
    flags: frozenset = frozenset()

    @property
    def declined(self) -> bool:
        return self.outcome == "declined"


@dataclass
class Trace:
    calls: list[Call] = field(default_factory=list)

    def record(self, call: Call) -> Call:
        self.calls.append(call)
        return call

    def of(self, tool: str) -> list[Call]:
        return [c for c in self.calls if c.tool == tool]

    @property
    def declines(self) -> list[Call]:
        return [c for c in self.calls if c.declined]

    @property
    def issues(self) -> list:
        """Every issue any tool reported, in the order they were reported."""
        out = []
        for call in self.calls:
            out.extend(call.issues)
        return out

    def counts(self) -> dict[str, int]:
        tally = {level: 0 for level in LEVELS}
        for issue in self.issues:
            tally[issue.level] += 1
        return tally

    def numbers(self) -> set[float]:
        pool: set[float] = set()
        for call in self.calls:
            pool |= set(call.numbers)
            for token in numbers_in_text(call.rendered):
                pool.add(float(token))
        return pool


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def check_refusals_terminal(trace: Trace) -> list[str]:
    """I3.  A refusal ends the attempt; it is not a starting position.

    The forbidden move is narrow and worth stating precisely: calling the
    same tool again, about the same thing, having changed only the knobs.
    Designing into a *different* vector after a refusal is not a retry, it
    is a different question, and the contract allows it -- which is why
    `target` and `tuned` are recorded separately.
    """
    broken: list[str] = []
    refused: dict[tuple, Call] = {}
    for call in trace.calls:
        key = (call.tool, call.target)
        if key in refused:
            first = refused[key]
            broken.append(
                f"{call.tool} declined with {first.declined_with} and was "
                f"called again on the same target "
                f"({first.tuned!r} -> {call.tuned!r})"
            )
        if call.declined:
            refused.setdefault(key, call)
    return broken


def check_issues_survive(trace: Trace, answer: str) -> list[str]:
    """I2.  Severity propagates, and never flattens.

    Two failures to catch, and they are different.  Dropping an issue is
    the obvious one.  Downgrading is the quiet one: nine issues become
    "design complete with some notes", every word of it true except the
    level.  So the check is per issue -- the text must be present, and it
    must be present under its own mark.
    """
    # Compared with whitespace collapsed, because several renderers wrap
    # issue text at 66 columns and put the level mark only on the first
    # line.  `textwrap.wrap` splits on whitespace and keeps the words, so
    # collapsing both sides reconstructs what was written -- and a check
    # that insisted on the unwrapped string would call every wrapped
    # issue a dropped one.
    flat_answer = " ".join(answer.split())
    broken: list[str] = []
    for issue in trace.issues:
        text = " ".join(issue.text.split())
        if text not in flat_answer:
            broken.append(f"[{issue.level}] dropped from the answer: {issue.text}")
            continue
        marked = f"{LEVEL_MARK[issue.level]} {text}"
        spelled = f"[{issue.level}] {text}"
        if marked not in flat_answer and spelled not in flat_answer:
            broken.append(
                f"{issue.text!r} appears without its {issue.level} mark"
            )
    return broken


def check_numbers_traceable(trace: Trace, answer: str) -> list[str]:
    """I1.  A model may describe a number; it may never supply one.

    Enforced as origin, not as arithmetic: every numeric token in the
    answer must be one the toolchain produced.  That leaves an orchestrator
    exactly two honest ways to write: quote a renderer verbatim, or write
    prose with no digits in it.  Both are fine.  Inventing a volume is not,
    and neither is restating a real number the model re-derived itself,
    because the two are indistinguishable from the outside.

    Rounding is allowed, at the precision the answer is written to, since
    that is what the renderers do with a melting temperature.
    """
    pool = trace.numbers()
    broken: list[str] = []
    for token in sorted(numbers_in_text(answer)):
        value = float(token)
        if "." in token:
            places = len(token.split(".", 1)[1])
            ok = any(round(n, places) == value for n in pool)
        else:
            ok = any(round(n) == value or int(n) == value for n in pool)
        if not ok:
            broken.append(f"{token} appears in the answer and in no tool result")
    return broken


def audit(trace: Trace, answer: str) -> dict[str, list[str]]:
    """Every invariant that can be checked after the fact, in one call."""
    return {
        "I1": check_numbers_traceable(trace, answer),
        "I2": check_issues_survive(trace, answer),
        "I3": check_refusals_terminal(trace),
    }
