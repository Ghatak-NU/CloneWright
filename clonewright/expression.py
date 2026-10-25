"""What the construct expresses, and whether E. coli is the right question.

`design` answers a DNA question: which enzymes, which primers, which frame.
This module answers the protein one -- what comes out of the cells, what
has to be added to the culture to get it, and which strain can make it at
all -- and it is the layer that makes this an *E. coli expression* tool
rather than a general cloning one.

Three things live here.

**The gate.** A vector that does not express in a bacterium is refused
before any primer is designed.  A CMV or GAL1 cassette is not a borderline
case to be warned about: no E. coli strain transcribes it, so primers for
cloning into it are primers for a different experiment, and designing them
quietly would be the worst outcome.

**The product.** The construct is translated from the start codon the
ribosome will actually use -- the vector's, for a fusion; the insert's, for
a native clone -- and what comes off is reported: its length, its mass, the
tags on each end, and where the stop codon falls.  This is the step that
catches a C-terminal tag that is never translated, which is invisible on
the DNA and obvious on the protein.

**The host.** Induction, antibiotics, and the T7 trap: a pET vector in
plain BL21 yields nothing at all while the plate looks completely normal.
That check lives in `strains`; this module is what asks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import design as design_mod
from . import source as source_mod
from . import strains as strains_mod
from .design import Design, Issue, VectorView
from .seqio import Feature, Record, translate

# Average residue masses in daltons, plus water for the finished chain.
# Standard values; printed in the report rather than only used, because
# like the codon tables they are transcribed rather than derived.
RESIDUE_MASS: dict[str, float] = {
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.10, "W": 186.21, "Y": 163.18, "V": 99.13,
}
WATER_MASS = 18.02

# Molar extinction coefficients at 280 nm, M-1 cm-1 (Pace et al. 1995).
# A protein with none of these three residues cannot be quantified by
# A280 at all, which is worth being told before the purification rather
# than at the spectrophotometer.
EXTINCTION = {"W": 5500.0, "Y": 1490.0, "C": 125.0}


@dataclass(frozen=True)
class Product:
    """The protein the finished construct encodes."""

    peptide: str
    # Residues contributed before and after the insert's own coding
    # sequence -- the tags, protease sites and linkers the vector adds.
    leader: str
    body: str
    trailer: str
    # Where translation starts and stops in the construct, 0-based.
    start: int
    stop: int | None
    stop_codon: str = ""
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.peptide)

    @property
    def mass_kda(self) -> float:
        total = sum(RESIDUE_MASS.get(r, 110.0) for r in self.peptide)
        return (total + WATER_MASS) / 1000.0

    @property
    def extinction(self) -> float:
        return sum(
            self.peptide.count(residue) * value
            for residue, value in EXTINCTION.items()
        )

    @property
    def a280_for_1mg_ml(self) -> float:
        """Absorbance of a 1 mg/mL solution in a 1 cm path."""
        mass = self.mass_kda * 1000.0
        return self.extinction / mass if mass else 0.0

    def describe(self) -> str:
        return f"{len(self)} residues, {self.mass_kda:.1f} kDa"


def product_of(d: Design) -> Product | None:
    """Translate the finished construct from the start codon that is used.

    Which start codon that is depends on the cloning mode, and getting it
    wrong gives a protein that is wrong in a plausible-looking way, so it
    is taken from the same function the design took it from rather than
    guessed at.

    `d.vector` is already the strand the cassette reads along -- `design`
    turned it round at the top -- which is exactly the precondition the
    frame-finding code was written for.
    """
    construct = d.construct
    n = len(construct)
    if not construct:
        return None

    backbone_len = len(d.backbone)
    # The core's offset inside the construct: the backbone, then whatever
    # the 5' end of the digested amplicon contributes before the insert's
    # own first base -- the half of the restriction site left after the
    # cut, plus any pad added to fix the frame.
    lead = d.forward.site[d.pair.five.top :] + d.forward.pad
    core_at = backbone_len + len(lead)

    if d.options.mode in ("nfusion",):
        frame = design_mod.fusion_frame(d.vector)
        if frame is None:
            return None
        cut3 = d.pair.pos_three + d.pair.three.top
        start = (frame.start - cut3) % len(d.vector)
        if start > backbone_len:
            return None
    else:
        # The insert brings its own start codon: the first ATG at or after
        # the point its own sequence begins.
        start = None
        for i in range(core_at, min(core_at + 60, n) - 2):
            if construct[i : i + 3] == "ATG":
                start = i
                break
        if start is None:
            return None

    # Translate forward from there, wrapping the circle once at most.
    runway = construct[start:] + construct[: start]
    peptide = translate(runway, to_stop=True)
    consumed = len(peptide) * 3
    truncated = consumed + 3 > len(runway)
    stop_at: int | None = None
    stop_codon = ""
    if not truncated:
        stop_codon = runway[consumed : consumed + 3]
        stop_at = (start + consumed) % n

    # Split the chain at the insert's own contribution.
    leader_len = max((core_at - start) // 3, 0)
    body_len = len(d.core) // 3
    leader = peptide[:leader_len]
    body = peptide[leader_len : leader_len + body_len]
    trailer = peptide[leader_len + body_len :]
    return Product(
        peptide=peptide,
        leader=leader,
        body=body,
        trailer=trailer,
        start=start,
        stop=stop_at,
        stop_codon=stop_codon,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def suits_ecoli(view: VectorView) -> list[Issue]:
    """Whether this vector expresses protein in E. coli at all.

    Returns error-level issues when it does not.  The wording says what
    the vector *is* for, because the commonest way to arrive here is to
    have picked up the right gene and the wrong plasmid, and a bare
    refusal does not help with that.
    """
    issues: list[Issue] = []
    host = view.host
    promoter = view.promoter

    if host == "bacterial":
        return issues

    if host in ("mammalian", "yeast"):
        driver = promoter.name if promoter is not None else "its promoter"
        issues.append(Issue(
            "error",
            f"this is a {host} expression vector: {driver} is not "
            "transcribed by any E. coli RNA polymerase, so the construct "
            "would be built and selected perfectly well and express "
            "nothing. clonewright designs for expression in E. coli; for "
            f"this vector the E. coli step is cloning only, and expression "
            f"happens after the construct reaches {host} cells",
        ))
        return issues

    # host is None: the annotations do not say.  That is a different
    # answer from "not bacterial", and it is recoverable -- the map may
    # simply be bare -- so it refuses with the way out attached.
    issues.append(Issue(
        "error",
        "nothing in this vector's annotations says it expresses in a "
        "bacterium: no ribosome binding site, no T7 or host promoter "
        "driving the cloning site, no named bacterial backbone. If the map "
        "is unannotated, `clonewright annotate` derives the cassette from "
        "the sequence; if it is annotated and this is still the answer, the "
        "vector is probably not a bacterial expression vector",
    ))
    return issues


# ---------------------------------------------------------------------------
# the whole expression picture
# ---------------------------------------------------------------------------


@dataclass
class ExpressionPlan:
    """How the construct is expressed, and what has to be true for it to be."""

    view: VectorView
    hosts: strains_mod.HostPlan
    spec: object | None = None
    product: Product | None = None
    assessment: object | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def plan(
    d: Design,
    *,
    view: VectorView | None = None,
    spec=None,
    origin: str = "",
    cloning: str = strains_mod.DEFAULT_CLONING,
    expression: str = strains_mod.DEFAULT_EXPRESSION,
) -> ExpressionPlan:
    """Everything about expressing a finished design, in one object.

    Takes the design rather than the vector, because half of what matters
    is a property of the construct and not of either molecule going in:
    which start codon is used, what the product weighs, whether the
    C-terminal tag survives the insert's stop codon.

    The strain is a *suggestion* derived from the insert -- rare codons
    point at Rosetta -- and the suggestion is then checked against the
    vector, which is the order that catches the T7 trap: picking a strain
    for the gene and then asking whether it can transcribe the plasmid.
    """
    if view is None:
        # `d.vector` is the strand the cassette reads along, which may be
        # the reverse complement of the map that went in.  `inspect` does
        # that flip itself, so it has to be handed the original.
        original = (
            d.vector.reverse_complement() if d.vector_flipped else d.vector
        )
        view = design_mod.inspect(original)
    issues: list[Issue] = []

    assessment = None
    rare = False
    if origin:
        assessment = source_mod.assess(origin, d.insert)
        issues.extend(getattr(assessment, "issues", []))
        rare = any(
            i.level in ("warning", "error") and "codon" in i.text.lower()
            for i in getattr(assessment, "issues", [])
        )

    # An explicitly named strain is the caller's decision and stands.
    # Otherwise the insert chooses: rare codons point at Rosetta, which
    # is a property of the gene, not of the vector.
    if expression and expression != strains_mod.DEFAULT_EXPRESSION:
        chosen = strains_mod.get(expression)
    else:
        chosen = strains_mod.suggest_expression_strain(rare_codons=rare)

    antibiotic = getattr(spec, "antibiotic", "")
    hosts = strains_mod.plan_hosts(
        view.reading,
        cloning=strains_mod.get(cloning),
        expression=chosen,
        vector_antibiotic=antibiotic,
    )
    issues.extend(hosts.issues)

    made = product_of(d)
    if made is not None:
        if made.truncated:
            issues.append(Issue(
                "warning",
                "translation from the start codon runs the whole way round "
                "the construct without reaching a stop codon. Either the "
                "insert has no stop and nothing downstream supplies one, or "
                "the frame is not what it is taken to be",
            ))
        elif made.trailer:
            issues.append(Issue(
                "note",
                f"the product carries {len(made.trailer)} residue(s) after "
                f"the insert's own sequence ({made.trailer}), contributed by "
                "the vector between the cloning site and the stop codon",
            ))
        if made.extinction == 0.0:
            issues.append(Issue(
                "warning",
                "the product contains no tryptophan, tyrosine or cysteine, "
                "so it does not absorb at 280 nm: concentration cannot be "
                "measured that way. Use a colourimetric assay (BCA or "
                "Bradford) instead",
            ))

    return ExpressionPlan(
        view=view,
        hosts=hosts,
        spec=spec,
        product=made,
        assessment=assessment,
        issues=issues,
    )
