"""Where the gene came from, and what that costs in E. coli.

Two facts about a gene's origin change how it has to be cloned and
expressed, and neither is visible in the vector or the primers.

**E. coli cannot splice.**  A gene amplified from the genomic DNA of an
organism whose genes carry introns arrives with those introns still in
it.  The PCR works, the ligation works, the colony PCR gives a band of
the expected size -- because the expected size was computed from the
same genomic sequence -- and the protein is nonsense.  The template has
to be cDNA, or a synthetic construct, and that is a decision made before
any of the rest of the protocol runs.

How much this matters depends entirely on the organism, which is why it
is asked rather than assumed.  Nearly every human gene has introns.
Only about one yeast gene in twenty does, so a fungal ORF amplified from
genomic DNA is usually fine -- *Candida albicans* rad26 is, which is why
the Rajan case never ran into this.  Treating those two the same would
mean either a false alarm on every fungal clone or silence on every
mammalian one.

The sequence can corroborate but never settle it.  An open reading frame
that runs cleanly from ATG to a stop, in frame, with no interruptions, is
what a cDNA looks like -- and also what a single-exon gene looks like.
An ORF that does *not* read cleanly is much stronger evidence: something
is wrong, and unspliced introns are the usual something.

**E. coli reads some codons slowly.**  A gene from an organism with a
different codon bias can be full of codons for which E. coli keeps very
little tRNA.  Scattered ones cost yield; tandem ones and ones near the
start codon cause ribosome stalling, truncation and frameshifting.  The
fix is a strain carrying the missing tRNAs, or a synthesised gene, and
knowing which before the first expression attempt saves a repeat.

Codon frequencies are transcribed from published E. coli K-12 usage
tables.  Like the enzyme buffer data, they are not derived from anything
in your sequences, so the figures are printed rather than only acted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .design import STOP_CODONS, Issue, OpenReadingFrame, read_orf
from .seqio import Record

# Codons E. coli uses rarely, with their approximate frequency per
# thousand in K-12, and whether Rosetta's pRARE plasmid supplies the
# tRNA.  These six are the classic set pRARE was built around; Rosetta 2
# adds CGG.
RARE_CODONS: dict[str, tuple[str, float, bool]] = {
    "AGA": ("Arg", 2.1, True),
    "AGG": ("Arg", 1.2, True),
    "CGA": ("Arg", 3.6, False),
    "CGG": ("Arg", 5.4, False),
    "ATA": ("Ile", 4.4, True),
    "CTA": ("Leu", 3.9, True),
    "CCC": ("Pro", 5.5, True),
    "GGA": ("Gly", 7.9, True),
}

# A rare codon in the first stretch of the message hurts most: the
# ribosome is still engaging and a stall there aborts the whole product
# rather than slowing it.
N_TERMINAL_CODONS = 25

# Fractions of the coding sequence above which the rare-codon load is
# worth acting on.
RARE_NOTEWORTHY = 0.05
RARE_SERIOUS = 0.08

# What an average E. coli gene carries of these same codons, from the
# frequencies above.  Without it the percentages mean nothing: 10% reads
# as small until you know the host's own genes sit near 3.4%.
ECOLI_BASELINE = sum(freq for _, freq, _ in RARE_CODONS.values()) / 1000.0


@dataclass(frozen=True)
class Origin:
    """An organism a gene can come from, and what that implies."""

    name: str
    label: str
    splices: bool
    intron_note: str
    codon_bias: str
    # "high", "low" or "none".  Nearly every mammalian gene has introns;
    # only about one fungal gene in twenty does.  Grading the two the
    # same would mean a false alarm on every fungal clone or silence on
    # every mammalian one.
    intron_risk: str = "none"

    @property
    def needs_cdna(self) -> bool:
        return self.splices


ORIGINS: dict[str, Origin] = {
    "mammalian": Origin(
        "mammalian", "mammalian",
        splices=True,
        intron_note="nearly every mammalian gene has introns, so genomic "
                    "DNA is almost never the right template",
        codon_bias="mammalian codon usage differs enough from E. coli that "
                   "rare-codon load is worth checking",
        intron_risk="high",
    ),
    "yeast": Origin(
        "yeast", "yeast or other fungus",
        splices=True,
        intron_note="only about one fungal gene in twenty has an intron, so "
                    "genomic DNA is often fine -- but check this gene rather "
                    "than the average",
        codon_bias="fungal codon usage is AT-leaning and can carry rare "
                   "arginine and isoleucine codons",
        intron_risk="low",
    ),
    "kinetoplastid": Origin(
        "kinetoplastid", "kinetoplastid protozoan (Leishmania, Trypanosoma)",
        splices=False,
        intron_note="trypanosomatids transcribe polycistronically and mature "
                    "their messages by trans-splicing, so cis-introns are "
                    "vanishingly rare -- a couple are known in the whole "
                    "genome -- and genomic DNA is the normal template",
        codon_bias="Leishmania and Trypanosoma genomes are GC-rich, so the "
                   "rare codons that turn up are the GC-rich arginines "
                   "rather than the AT-rich set a fungal gene brings",
        intron_risk="low",
    ),
    "apicomplexan": Origin(
        "apicomplexan", "apicomplexan protozoan (Plasmodium, Toxoplasma)",
        splices=True,
        intron_note="apicomplexan genes commonly carry introns, so genomic "
                    "DNA is usually the wrong template",
        codon_bias="Plasmodium is extremely AT-rich, around 80%, which makes "
                   "for a heavy rare-codon load and low-complexity runs that "
                   "are hard to synthesise as well as to express; "
                   "Toxoplasma is more balanced",
        intron_risk="high",
    ),
    "viral": Origin(
        "viral", "viral",
        splices=False,
        intron_note="most viral ORFs are uninterrupted, though retroviruses, "
                    "herpesviruses and adenoviruses do splice; a cloned viral "
                    "CDS is usually already intronless",
        codon_bias="viral codon usage follows the host the virus replicates "
                   "in, so check rather than assume",
        intron_risk="low",
    ),
    "bacterial": Origin(
        "bacterial", "bacterial",
        splices=False,
        intron_note="bacterial genes have no introns, so genomic DNA is a "
                    "perfectly good template",
        codon_bias="another bacterium can still have a very different bias; "
                   "GC-rich genomes are the usual source of trouble",
    ),
    "synthetic": Origin(
        "synthetic", "synthetic or codon-optimised",
        splices=False,
        intron_note="a synthesised gene has no introns by construction",
        codon_bias="if it was optimised for E. coli the rare-codon load "
                   "should already be low; this checks that it was",
    ),
}


# What people call these organisms, mapped to the clade whose properties
# actually decide the answer.  Nobody types "apicomplexan" at a command
# line; they type "human", "yeast", or "plasmodium".  The table is here
# rather than in the command-line code because every caller needs it --
# the sentence planner and the `--origin` flag both reach `get`, and an
# alias known to only one of them is a crash waiting in the other.
ALIASES: dict[str, str] = {
    "human": "mammalian", "homo sapiens": "mammalian", "hsapiens": "mammalian",
    "mouse": "mammalian", "mus musculus": "mammalian", "rat": "mammalian",
    "cho": "mammalian", "hek293": "mammalian",
    "s. cerevisiae": "yeast", "cerevisiae": "yeast", "saccharomyces": "yeast",
    "candida": "yeast", "albicans": "yeast", "pichia": "yeast",
    "aspergillus": "yeast", "neurospora": "yeast", "fungal": "yeast",
    "leishmania": "kinetoplastid", "trypanosoma": "kinetoplastid",
    "plasmodium": "apicomplexan", "toxoplasma": "apicomplexan",
    "e. coli": "bacterial", "e.coli": "bacterial", "ecoli": "bacterial",
    "mycobacterium": "bacterial", "bacillus": "bacterial",
    "codon optimised": "synthetic", "codon optimized": "synthetic",
    "gene synthesis": "synthetic", "idt": "synthetic", "twist": "synthetic",
}


def get(name: str) -> Origin:
    key = name.strip().lower()
    if key in ORIGINS:
        return ORIGINS[key]
    if key in ALIASES:
        return ORIGINS[ALIASES[key]]
    known = ", ".join(sorted(ORIGINS))
    raise KeyError(
        f"unknown source organism {name!r}; known: {known} "
        f"(or a species: {', '.join(sorted(ALIASES)[:6])}, ...)"
    )


# ---------------------------------------------------------------------------
# does this read like a coding sequence?
# ---------------------------------------------------------------------------


def template_guidance(origin: Origin, insert: Record) -> list[Issue]:
    """What the gene's origin says about which template to amplify from."""
    orf = read_orf(insert.seq)
    out: list[Issue] = []

    if not orf.is_clean:
        level = "error" if orf.internal_stops or not orf.in_frame else "warning"
        reason = "; ".join(orf.faults())
        extra = (
            " Unspliced introns are the usual cause when the sequence came "
            "from genomic DNA."
            if origin.splices
            else ""
        )
        out.append(
            Issue(
                level,
                f"this does not read as an uninterrupted coding sequence: "
                f"{reason}.{extra} Expressing it in E. coli will not give "
                "the intended protein",
            )
        )
        return out

    if origin.intron_risk == "high":
        out.append(
            Issue(
                "warning",
                f"E. coli cannot splice, and {origin.intron_note}. This "
                "sequence does read as a clean ORF, which is what a cDNA "
                "looks like -- but it is also what a single-exon gene looks "
                "like, so confirm the template is cDNA or a synthetic "
                "construct rather than genomic DNA",
            )
        )
    elif origin.intron_risk == "low":
        out.append(
            Issue(
                "note",
                f"E. coli cannot splice, but {origin.intron_note}. This "
                "sequence reads as a clean ORF, so genomic DNA is a "
                "reasonable template here unless this gene is one of the "
                "spliced minority",
            )
        )
    else:
        out.append(
            Issue(
                "note",
                f"{origin.intron_note}, and the sequence reads as a clean "
                "ORF, so splicing is not a concern here",
            )
        )
    return out


# ---------------------------------------------------------------------------
# codon usage
# ---------------------------------------------------------------------------


@dataclass
class RareCodonReport:
    total: int
    counts: dict[str, int] = field(default_factory=dict)
    runs: list[tuple[int, int, str]] = field(default_factory=list)
    n_terminal: list[tuple[int, str]] = field(default_factory=list)

    @property
    def rare(self) -> int:
        return sum(self.counts.values())

    @property
    def fraction(self) -> float:
        return self.rare / self.total if self.total else 0.0

    @property
    def unsupplied(self) -> list[str]:
        """Rare codons here that Rosetta's pRARE does not cover."""
        return sorted(
            c for c in self.counts if not RARE_CODONS[c][2] and self.counts[c]
        )


def rare_codon_report(seq: str) -> RareCodonReport:
    """Count the codons E. coli is short of tRNA for.

    Three things are counted separately because they do different
    damage: the overall load, which costs yield; tandem runs, where a
    stalled ribosome is likely to be hit by the one behind it; and
    codons near the start, where a stall aborts the product instead of
    slowing it.
    """
    seq = seq.upper()
    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    report = RareCodonReport(total=len(codons))
    for codon in codons:
        if codon in RARE_CODONS:
            report.counts[codon] = report.counts.get(codon, 0) + 1

    run_start = None
    for index, codon in enumerate(codons + [""]):
        if codon in RARE_CODONS:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= 2:
                report.runs.append(
                    (run_start + 1, index - run_start,
                     "-".join(codons[run_start:index]))
                )
            run_start = None

    report.n_terminal = [
        (index + 1, codon)
        for index, codon in enumerate(codons[:N_TERMINAL_CODONS])
        if codon in RARE_CODONS
    ]
    return report


def codon_guidance(origin: Origin, insert: Record) -> list[Issue]:
    report = rare_codon_report(insert.seq)
    out: list[Issue] = []
    if not report.total:
        return out

    worst = sorted(
        report.counts.items(), key=lambda kv: -kv[1]
    )[:4]
    breakdown = ", ".join(
        f"{codon} ({RARE_CODONS[codon][0]}) x{count}" for codon, count in worst
    )
    headline = (
        f"{report.rare} of {report.total} codons ({report.fraction:.1%}, "
        f"against about {ECOLI_BASELINE:.1%} in an average E. coli gene) are "
        f"ones E. coli keeps little tRNA for: {breakdown}"
    )

    if report.fraction >= RARE_SERIOUS:
        out.append(
            Issue(
                "warning",
                headline + ". That is a high enough load to cost yield and "
                "risk truncation; a strain supplying the missing tRNAs, such "
                "as Rosetta, is the cheaper fix than resynthesising the gene",
            )
        )
    elif report.fraction >= RARE_NOTEWORTHY:
        out.append(Issue("note", headline + ", which is a moderate load"))
    else:
        out.append(Issue("note", headline + ", which is low"))

    if report.runs:
        shown = ", ".join(
            f"{seq} at codon {at}" for at, _, seq in report.runs[:3]
        )
        out.append(
            Issue(
                "warning",
                f"{len(report.runs)} run(s) of consecutive rare codons: "
                f"{shown}. Tandem rare codons stall a ribosome long enough "
                "for the next one to catch it, which is where frameshifting "
                "and truncation come from, and they matter more than the "
                "same codons spread out",
            )
        )
    if report.n_terminal:
        shown = ", ".join(
            f"{codon} at codon {at}" for at, codon in report.n_terminal[:4]
        )
        out.append(
            Issue(
                "warning",
                f"rare codons in the first {N_TERMINAL_CODONS} codons: "
                f"{shown}. A stall this early aborts the product rather than "
                "slowing it, so these cost more than their number suggests",
            )
        )
    if report.unsupplied:
        out.append(
            Issue(
                "note",
                "pRARE does not supply tRNA for "
                + ", ".join(report.unsupplied)
                + ", so Rosetta will not fix those; Rosetta 2 adds CGG",
            )
        )
    out.append(
        Issue(
            "note",
            "codon frequencies are transcribed from published E. coli K-12 "
            "usage tables, not derived from your sequences",
        )
    )
    return out


# ---------------------------------------------------------------------------
# putting it together
# ---------------------------------------------------------------------------


@dataclass
class SourceAssessment:
    origin: Origin
    insert: Record
    orf: OpenReadingFrame
    codons: RareCodonReport
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def wants_rare_codon_strain(self) -> bool:
        return (
            self.codons.fraction >= RARE_SERIOUS
            or bool(self.codons.runs)
            or len(self.codons.n_terminal) >= 2
        )


def assess(origin: Origin | str, insert: Record) -> SourceAssessment:
    """Everything the gene's origin implies for expressing it in E. coli."""
    origin = get(origin) if isinstance(origin, str) else origin
    assessment = SourceAssessment(
        origin=origin,
        insert=insert,
        orf=read_orf(insert.seq),
        codons=rare_codon_report(insert.seq),
    )
    assessment.issues.extend(template_guidance(origin, insert))
    assessment.issues.extend(codon_guidance(origin, insert))
    return assessment
