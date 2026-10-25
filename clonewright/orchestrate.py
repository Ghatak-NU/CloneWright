"""Question to audited answer.  Picks tools, carries provenance, computes nothing.

Every number in anything this module returns was put there by a function
with a test behind it, and `assemble` refuses to hand back an answer where
that is not true.

The split that makes an agent testable: **the route is code and only the
judgement is a model.**  A `Planner` decides what the user wants and a
`Narrator` writes the prose around the result; everything between them --
which tools, in what order, what happens when one declines -- is the fixed
pipeline below, and it all goes through `agent.Session`, so the gate applies
whoever is driving.

`StandardRoute` is the default planner and is wholly deterministic, which
means this orchestrator can be run, tested and used with no model present
at all.  That is not a stub for a model to replace later; it is the shape
the contract asks for, with a seam where judgement belongs.  `planner.py`
is what fills the seam.

What a model gets to decide: what was asked for, and how to say it.  What
it does not: anything in `run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol as _Protocol

from . import design as design_mod
from . import expression as expression_mod
from . import pcr
from . import report
from . import strains as strains_mod
from . import vectors as vectors_mod
from . import verify as verify_mod
from .agent import Session
from .trace import InvariantError, Trace, audit


@dataclass(frozen=True)
class Request:
    """One cloning question, in the only terms the toolchain understands.

    Turning *"put SOD1 into pET-28a and tell me how to check it"* into this
    is the one judgement at the front of the pipeline, and the only part of
    the input a model is trusted with.  Everything here is a path, a name
    from a fixed set, or a flag: nothing a model fills in can become a
    number in the answer.
    """

    insert: str
    # Empty means "choose one", which is answered by designing into every
    # bundled vector rather than by preference.
    vector: str = ""
    mode: str = "auto"
    # The organism the gene came from.  Off unless given, because the two
    # things it decides -- whether the template may carry introns, and
    # whether the codon usage needs a tRNA-supplementing strain -- cannot
    # be guessed from the sequence.
    origin: str = ""
    cloning_strain: str = strains_mod.DEFAULT_CLONING
    expression_strain: str = strains_mod.DEFAULT_EXPRESSION
    verify: bool = True
    # Show the whole vector comparison even when a vector was named.
    compare: bool = False
    screening_polymerase: str = "taq"


@dataclass
class Outcome:
    """What came back, and the record of how."""

    request: Request
    trace: Trace
    blocks: list[str] = field(default_factory=list)
    answer: str = ""
    refused: bool = False
    refusal: str = ""
    design: object | None = None
    view: object | None = None
    plan: object | None = None
    spec: object | None = None
    candidates: tuple = ()

    @property
    def issues(self) -> list:
        return self.trace.issues


@dataclass(frozen=True)
class Summary:
    """What a narrator is told: objects, never a rendering of one.

    `report.render*` output is terminal and does not come back in here.
    Prose summarisation at a seam is where information dies, and a narrator
    handed a paragraph instead of a `Design` could not count the issues
    even if it wanted to.
    """

    request: Request
    view: object | None
    design: object | None
    plan: object | None
    refused: bool
    refusal: str
    issues: tuple


class Planner(_Protocol):
    def route(self, request: Request) -> Request:
        """Settle anything the request left open.  May not add numbers."""


Narrator = Callable[[Summary], str]


class StandardRoute:
    """The default planner, which settles nothing.

    It looks empty because it is, and that is the honest default rather
    than a placeholder.  Everything a planner could decide from the
    sequences -- the cloning mode, the host, the driving promoter, which
    cassette the insert fuses to -- is already decided inside Layer 1 by
    code with tests behind it, and `mode="auto"` reaches all of it.  A
    planner that second-guessed any of that would be a model overruling
    arithmetic, which is the one thing the contract forbids outright.

    What a real planner is for is the step before this: turning a sentence
    into a `Request`.  That judgement has no home in code, and when a model
    does it, this is the interface it implements.
    """

    def route(self, request: Request) -> Request:
        return request


def plain_narration(summary: Summary) -> str:
    """The default narrator.  Writes no digits, so it cannot break I1.

    Deliberately thin.  The renderers already say everything that can be
    said precisely, and a default that paraphrased them would be competing
    with the part of the system that is actually checked.
    """
    if summary.refused:
        return (
            "This design was refused, and the reasons are above. A refusal "
            "is a result: nothing here can be loosened to make it pass, and "
            "the next step is a different insert, a different vector, or a "
            "different cloning strategy."
        )
    levels = {i.level for i in summary.issues}
    if "warning" in levels:
        return (
            "The design succeeded, with warnings. Read them before ordering: "
            "a design that returns a warning has not simply succeeded, and "
            "every one above is something a bench step will or will not "
            "forgive."
        )
    return (
        "The design succeeded. The notes above are the reasoning, not "
        "problems; read the verification panel before ordering oligos."
    )


def assemble(session: Session, blocks: list[str], prose: str) -> str:
    """Put the answer together, and refuse to return a bad one.

    This is where I1 and I2 stop being descriptions and start being
    enforced.  The audit runs on the finished text, and a violation raises
    rather than returning: an answer that drops an issue or carries a
    number no tool produced is not a flawed answer to be flagged, it is not
    an answer.

    The consequence for anything calling this is a useful discipline --
    whatever you compute, you render.  Issues live on results, the audit
    knows every result that was returned, and the only way to satisfy it is
    to show them.
    """
    parts = [b for b in blocks if b]
    if prose:
        parts.append(prose)
    answer = "\n\n".join(parts)

    broken = {k: v for k, v in audit(session.trace, answer).items() if v}
    if broken:
        many = len(broken) > 1
        detail = "; ".join(
            (f"{name}: " if many else "")
            + reasons[0]
            + (f" (and {len(reasons) - 1} more)" if len(reasons) > 1 else "")
            for name, reasons in sorted(broken.items())
        )
        raise InvariantError(", ".join(sorted(broken)), detail)
    return answer


class Orchestrator:
    """One run of the toolchain, from a request to an audited answer."""

    def __init__(
        self,
        planner: Planner | None = None,
        narrator: Narrator | None = None,
        session: Session | None = None,
    ):
        self.planner = planner or StandardRoute()
        self.narrator = narrator or plain_narration
        self.session = session or Session()

    def run(self, request: Request) -> Outcome:
        s = self.session
        request = self.planner.route(request)
        out = Outcome(request=request, trace=s.trace)

        insert = s.call("seqio.load", path=request.insert)
        opts = design_mod.Options(mode=request.mode)

        name = request.vector
        if not name or request.compare:
            usable, rejected = s.call("vectors.shortlist", insert=insert,
                                      opts=opts)
            out.candidates = tuple(usable)
            out.blocks.append(s.call("report.render_shortlist",
                                     usable=usable, rejected=rejected))
            if not name:
                if not usable:
                    out.refused = True
                    out.refusal = (
                        "no bundled vector can take this insert"
                    )
                    out.answer = assemble(s, out.blocks, self._narrate(out))
                    return out
                # The agent's own choice, and it is the ranking's choice:
                # fewest warnings, then least vector sequence removed.
                name = usable[0].spec.name

        vector, spec = s.call("vectors.resolve", name=name)
        out.spec = spec

        # Read the vector on the strand its cassette uses, before designing
        # into it.  `inspect` is the only honest way to ask: the functions
        # it composes answer about the wrong strand when the cassette reads
        # against the map.
        view = s.call("design.inspect", vector=vector)
        out.view = view

        # The gate.  A vector that does not express in a bacterium is
        # refused here, before any primer exists, because primers for
        # cloning into it would answer a different question than the one
        # asked -- and would look exactly like an answer to this one.
        gate = s.call("expression.suits_ecoli", view=view)
        if any(i.level == "error" for i in gate):
            out.refused = True
            out.refusal = "; ".join(i.text for i in gate)
            out.blocks.append(_issue_only_block(gate, "not an E. coli "
                                                      "expression vector"))
            out.answer = assemble(s, out.blocks, self._narrate(out))
            return out

        try:
            d = s.call("design.design", insert=insert, vector=vector,
                       opts=opts)
        except design_mod.DesignFailed as exc:
            # I3.  Report it, explain it, stop.  There is no branch here
            # that tries again with the knobs moved, and `Session` would
            # refuse one if there were.
            out.refused = True
            out.refusal = str(exc)
            out.blocks.append(_refusal_block(exc))
            out.answer = assemble(s, out.blocks, self._narrate(out))
            return out

        out.design = d
        out.blocks.append(s.call("report.render", d=d))

        if not d.ok:
            # The commoner refusal: nothing was raised, the result came back
            # carrying error-level issues.  Building verification on top of
            # a design that failed its own checks would be ceremony.
            out.refused = True
            out.refusal = "; ".join(i.text for i in d.errors)
            out.answer = assemble(s, out.blocks, self._narrate(out))
            return out

        if request.origin:
            assessment = s.call("source.assess", origin=request.origin,
                                insert=insert)
            out.blocks.append(s.call("report.render_source",
                                     assessment=assessment))

        plan = s.call(
            "expression.plan",
            d=d,
            view=view,
            spec=spec,
            origin=request.origin,
            cloning=request.cloning_strain,
            expression=request.expression_strain,
        )
        out.plan = plan
        out.blocks.append(s.call("report.render_expression", plan=plan))

        if request.verify:
            construct, target = s.call("verify.construct_of", design=d)
            spec_v = verify_mod.PrimerSpec(conditions=opts.conditions)
            colony = s.call(
                "verify.colony_pcr_plan",
                construct=construct, target=target, parental=vector,
                spec=spec_v,
                polymerase=pcr.POLYMERASES[request.screening_polymerase],
            )
            sequencing = s.call(
                "verify.sequencing_plan",
                construct=construct, target=target, spec=spec_v,
            )
            out.blocks.append(s.call("report.render_colony", plan=colony))
            out.blocks.append(s.call("report.render_sequencing",
                                     plan=sequencing))

        out.answer = assemble(s, out.blocks, self._narrate(out))
        return out

    def _narrate(self, out: Outcome) -> str:
        return self.narrator(Summary(
            request=out.request,
            view=out.view,
            design=out.design,
            plan=out.plan,
            refused=out.refused,
            refusal=out.refusal,
            issues=tuple(out.trace.issues),
        ))


def _issue_only_block(issues, what: str) -> str:
    heading = f"-- {what} "
    lines = [heading + "-" * max(74 - len(heading), 0)]
    lines.extend(report._issue_block(issues))
    return "\n".join(lines)


def _refusal_block(exc: Exception, what: str = "design") -> str:
    """A raised refusal, shown the way a returned one is shown.

    `report` has no renderer for an exception, and giving it one would put
    another caller in front of the issue formatting.  This reuses the block
    the renderers already use, so a refusal that was raised and a refusal
    that was returned read the same and audit the same.
    """
    heading = f"-- {what} refused "
    lines = [heading + "-" * max(74 - len(heading), 0)]
    issues = getattr(exc, "issues", None)
    if issues:
        lines.extend(report._issue_block(issues))
    else:
        for index, chunk in enumerate(report._wrap(str(exc), 66)):
            mark = report.LEVEL_MARK["error"] if index == 0 else ""
            lines.append(f"  {mark:>5}  {chunk}")
    return "\n".join(lines)


def run(request: Request, **kwargs) -> Outcome:
    """Convenience: one request, one outcome, a fresh session."""
    return Orchestrator(**kwargs).run(request)
