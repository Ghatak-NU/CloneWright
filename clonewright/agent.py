"""The toolchain as a model is allowed to see it.

This is the part that makes clonewright an agent rather than a script: a
fixed registry of tools, a session that records every call, and a gate that
stops the one move a language model reliably makes when a tool says no --
asking again with the knobs moved.

`Session` is the only way in.  It forwards each call untouched -- no
defaults of its own, no retries, no massaging of arguments -- records what
happened, and enforces **I3**: a refusal is terminal, so the same question
cannot be re-asked with different options.  I1 and I2 are properties of the
finished answer rather than of a call, so they are audited afterwards by
`trace.audit`.

Nothing here computes.  Every number that reaches an answer originates in
the module being called, which is what gives the audit something real to
check.  A model drives the registry; it does not replace any part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

from . import (
    cassette,
    construct,
    design,
    enzymes,
    expression,
    ncbi,
    pcr,
    report,
    seqio,
    source,
    strains,
    vectors,
    verify,
)
from .trace import Call, InvariantError, Trace, issues_in, numbers_in


@dataclass(frozen=True)
class Tool:
    """One row of the contract."""

    name: str
    layer: int
    call: Callable[..., Any]
    declines: tuple[type[BaseException], ...] = ()
    # Arguments a retry would mutate.  Kept out of the identity of the
    # question so that I3 can tell "ask again, but louder" from "ask about
    # something else".
    tunable: frozenset[str] = frozenset()
    renders: bool = False
    # One line, shown to a model that is choosing a tool.
    summary: str = ""


def _rows() -> list[Tool]:
    T = Tool
    return [
        # -- Layer 0: bytes in ------------------------------------------
        T("seqio.load", 0, seqio.load, (seqio.SequenceError,),
          summary="read a molecule from FASTA, GenBank or SnapGene .dna"),
        T("seqio.load_sequence", 0, seqio.load_sequence,
          (seqio.SequenceError,),
          summary="the same, or a bare sequence pasted in as text"),
        T("vectors.resolve", 0, vectors.resolve, (vectors.VectorError,),
          summary="a bundled E. coli expression vector by name, or any map "
                  "by path"),
        T("ncbi.resolve", 0, ncbi.resolve, (ncbi.FetchError,),
          tunable=frozenset({"allow_network", "cache", "timeout", "email"}),
          summary="fetch an NCBI link or accession and splice the coding "
                  "sequence out of it"),
        T("cassette.annotate", 0, cassette.annotate,
          tunable=frozenset({"hints", "min_orf_bp"}),
          summary="derive the expression cassette of an unannotated vector "
                  "from its sequence"),

        # -- Layer 1: computation. A model may call any of it, and may not
        # reimplement, approximate or second-guess any of it.
        #
        # Vector reading is `inspect` and not the seven functions it
        # composes.  Those were written for a cassette that reads forward,
        # because `design` turns the vector round once at the top; called
        # directly on pET-28a they answer about the wrong strand and do it
        # quietly -- the frame anchored on KanR, the mode native.  A tool
        # with an unstated precondition is a trap, and the registry is not
        # the place to hand one to a model.
        T("design.inspect", 1, design.inspect,
          summary="read a vector on the strand its cassette uses"),
        T("expression.suits_ecoli", 1, expression.suits_ecoli,
          summary="whether this vector expresses protein in E. coli at all"),
        T("design.design", 1, design.design, (design.DesignFailed,),
          tunable=frozenset({"opts"}),
          summary="the enzyme pair, the primers and the reading frame"),
        T("design.validate", 1, design.validate,
          summary="re-check a finished design against both molecules"),
        T("design.collateral_losses", 1, design.collateral_losses,
          summary="vector features the excision removes"),
        T("design.primer_alternatives", 1, design.primer_alternatives,
          tunable=frozenset({"limit", "vary_pad", "vary_spacer"}),
          summary="other annealing regions, trading Tm against folding"),
        T("enzymes.plan_digest", 1, enzymes.plan_digest, (enzymes.EnzymeError,),
          summary="buffer, temperature and methylation for the double digest"),
        T("pcr.simulate", 1, pcr.simulate,
          summary="in-silico PCR, including mispriming"),
        T("source.assess", 1, source.assess,
          summary="introns and rare codons, by the organism the gene came "
                  "from"),
        T("strains.plan_hosts", 1, strains.plan_hosts,
          summary="which E. coli does which half, and the T7 trap"),
        T("strains.suggest_expression_strain", 1,
          strains.suggest_expression_strain,
          summary="the expression strain the insert's own properties point to"),
        T("expression.product_of", 1, expression.product_of,
          summary="translate the finished construct: the protein, its mass "
                  "and its tags"),
        T("expression.plan", 1, expression.plan,
          tunable=frozenset({"cloning", "expression", "origin"}),
          summary="product, induction, antibiotics and strain for a design"),
        T("vectors.shortlist", 1, vectors.shortlist,
          tunable=frozenset({"names", "opts"}),
          summary="design into every bundled vector and rank what worked"),
        T("construct.assemble", 1, construct.assemble,
          tunable=frozenset({"name"}),
          summary="the ligated plasmid with the features that survived, "
                  "ready to open in SnapGene"),

        # Validation is code only.  A probabilistic referee is not a
        # referee: the primers that prove the construct cannot come from
        # the same judgement that chose the primers being proved.
        T("verify.construct_of", 1, verify.construct_of,
          summary="the finished plasmid, and where the insert sits in it"),
        T("verify.sequencing_plan", 1, verify.sequencing_plan,
          tunable=frozenset({"geometry", "spec", "junction_flank", "strands"}),
          summary="Sanger primers that cover the insert and both junctions"),
        T("verify.colony_pcr_plan", 1, verify.colony_pcr_plan,
          tunable=frozenset({"spec", "polymerase", "anchor_distance",
                             "max_screen_bp"}),
          summary="colony screen, and whether a band can only come from a "
                  "real clone"),

        # -- Layer 4: rendering is terminal.  Its output goes to a human
        # and never becomes another tool's input.
        T("report.render", 4, report.render, renders=True,
          summary="the design, as a bench-ready report"),
        T("report.render_fetch", 4, report.render_fetch, renders=True,
          summary="what came back from NCBI, and whether the splice checks "
                  "out"),
        T("report.render_expression", 4, report.render_expression, renders=True,
          summary="the protein, the induction and the strain"),
        T("report.render_colony", 4, report.render_colony, renders=True,
          summary="the colony screen"),
        T("report.render_sequencing", 4, report.render_sequencing, renders=True,
          summary="the Sanger panel"),
        T("report.render_shortlist", 4, report.render_shortlist, renders=True,
          summary="the vector comparison"),
        T("report.render_source", 4, report.render_source, renders=True,
          summary="introns and rare codons"),
    ]


REGISTRY: dict[str, Tool] = {t.name: t for t in _rows()}


def catalogue() -> str:
    """The registry as a model should be shown it: name, layer, one line."""
    width = max(len(name) for name in REGISTRY)
    return "\n".join(
        f"  {t.name:<{width}}  L{t.layer}  {t.summary}"
        for t in REGISTRY.values()
    )


def _fingerprint(obj: Any, depth: int = 0) -> Any:
    """A stable identity for an argument.

    Keying on `id()` would let an orchestrator slip past I3 by reloading
    the same vector from the same file and calling the result a new
    question, so a `Record` is fingerprinted by its contents instead:
    read it twice and you get the same answer, change it and you do not.

    Contents means all of them.  Name and length alone looked frugal and
    were wrong -- two entirely different molecules of the same length
    under the same name fingerprint identically, and so does a vector
    with its annotations stripped, which is a different design and not the
    same question asked twice.  The sequence hash is what makes this an
    identity rather than a guess, and the feature layout is in it because
    the annotations decide the cassette and the polylinker.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if depth > 6:
        return type(obj).__name__
    seq = getattr(obj, "seq", None)
    name = getattr(obj, "name", None)
    if isinstance(seq, str) and isinstance(name, str):
        features = getattr(obj, "features", None) or ()
        return (
            "Record", name, len(seq), hash(seq),
            hash(tuple((f.name, f.start, f.end) for f in features)),
        )
    if isinstance(obj, (list, tuple)):
        return tuple(_fingerprint(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted(
            (str(k), _fingerprint(v, depth + 1)) for k, v in obj.items()
        ))
    if is_dataclass(obj) and not isinstance(obj, type):
        return (type(obj).__name__, tuple(
            (f.name, _fingerprint(getattr(obj, f.name, None), depth + 1))
            for f in fields(obj)
        ))
    return type(obj).__name__


def _split(tool: Tool, kwargs: dict) -> tuple[tuple, tuple]:
    """The call, as (what was asked, what it was asked with)."""
    target = tuple(sorted(
        (k, _fingerprint(v)) for k, v in kwargs.items()
        if k not in tool.tunable
    ))
    tuned = tuple(sorted(
        (k, _fingerprint(v)) for k, v in kwargs.items() if k in tool.tunable
    ))
    return target, tuned


class Session:
    """A run of the toolchain, watched.

    Hold one per user request.  Call tools through it by name; read
    `session.trace` afterwards and hand it to `trace.audit` along with
    whatever the orchestrator decided to say.
    """

    def __init__(self, registry: dict[str, Tool] | None = None,
                 trace: Trace | None = None):
        self.registry = REGISTRY if registry is None else registry
        self.trace = trace or Trace()

    def _check_terminal(self, tool: Tool, target: tuple, tuned: tuple) -> None:
        for call in self.trace.calls:
            if call.tool != tool.name or not call.declined:
                continue
            if call.target != target:
                continue
            raise InvariantError(
                "I3",
                f"{tool.name} already declined this with "
                f"{call.declined_with}; a refusal ends the attempt. Report "
                f"it, explain it, or ask about a different input -- do not "
                f"call it again with {tuned!r} in place of {call.tuned!r}",
            )

    def call(self, name: str, /, **kwargs):
        # Positional-only, because the tools' own arguments share this
        # namespace: `vectors.resolve(name=...)` takes a `name` too, and
        # with an ordinary parameter the forwarding call collided with
        # itself rather than reaching the tool.
        tool = self.registry.get(name)
        if tool is None:
            raise InvariantError(
                "contract",
                f"{name!r} is not in the registry; the tools are "
                f"{', '.join(sorted(self.registry))}",
            )

        target, tuned = _split(tool, kwargs)
        self._check_terminal(tool, target, tuned)

        try:
            result = tool.call(**kwargs)
        except tool.declines as exc:
            # A refusal is a result, and its reasons are part of the answer:
            # `DesignFailed` carries the issues that caused it, and I2
            # covers those exactly as it covers a warning on a design that
            # succeeded.
            self.trace.record(Call(
                tool=name,
                layer=tool.layer,
                target=target,
                tuned=tuned,
                outcome="declined",
                declined_with=type(exc).__name__,
                issues=issues_in(exc),
                # The message as well as the fields.  An exception's `args`
                # are not in its `__dict__`, so walking the object alone
                # misses everything a refusal actually says -- and a
                # refusal's text is the part most likely to be quoted,
                # numbers and all.
                numbers=frozenset(numbers_in(exc) | numbers_in(str(exc))),
            ))
            raise

        issues = issues_in(result)
        errors = [i for i in issues if i.level == "error"]

        self.trace.record(Call(
            tool=name,
            layer=tool.layer,
            target=target,
            tuned=tuned,
            # A refusal does not always arrive as an exception, and the
            # commoner shape is this one: a returned object carrying
            # error-level issues.  I3 is about not routing around a
            # refusal, so it has to recognise the one the tool actually
            # produces, or it guards the rare case and misses the real one.
            outcome="declined" if errors else "returned",
            declined_with=(
                f"{len(errors)} error-level issue(s)" if errors else ""
            ),
            issues=issues,
            numbers=frozenset(numbers_in(result)),
            rendered=result if tool.renders and isinstance(result, str) else "",
        ))
        return result

    def rendered(self) -> list[str]:
        """Every block a renderer produced, in order.

        The honest raw material for an answer: an orchestrator that pastes
        these and writes digit-free prose around them cannot break I1 or I2.
        """
        return [c.rendered for c in self.trace.calls if c.rendered]
