"""Verifying the construct: sequencing and colony-PCR primers.

Designing the cloning primers is half the job.  What comes back from the
ligation still has to be shown to be what was intended, and that is two
different assays needing two different kinds of primer:

* **Colony PCR** asks whether a colony carries the insert at all.  It is
  quick, runs on many colonies from crude lysate, and is a screen -- it
  says nothing about whether the sequence is right.
* **Sanger sequencing** asks whether the insert that is there is correct,
  base by base, including the two junctions, which are exactly where the
  cloning can have gone wrong.

Four things about these primers are easy to get wrong and are enforced
here rather than left to the user.

**A colony-PCR primer pair must not sit wholly inside the insert.**  The
ligation carries over residual template -- the parental plasmid the insert
was amplified from, and the PCR product itself -- and some of it survives
transformation.  An insert-internal pair amplifies all of that just as
happily as it amplifies a real recombinant, so every colony looks
positive.  Anchoring one primer in the vector backbone makes the product
conditional on the insert actually being joined to *this* vector, which is
the question being asked.  The same arrangement reports orientation for
free: a backwards insert gives no product at all, where a pair of flanking
vector primers would give a band of the same size either way.

**A sequencing primer has to sit back from what it is meant to read.**
The first bases after the primer are unreadable -- the dye front and the
small fragments are not resolved -- so a primer placed at the junction
reads everything *except* the junction.  Primers are therefore offset
upstream by the dead zone plus a margin.

**Tiled reads have to overlap.**  Spacing primers by the full read length
leaves each join between reads covered only by the ends of two traces,
which is the lowest-quality part of both.  The step is the usable length
minus the dead zone minus a required overlap, so consecutive reads agree
over a stretch that is good in both.

**A repeat blocks priming, not reading.**  A primer whose 3' end sits in a
sequence that occurs twice has two places to bind and gives a
superimposed trace.  The region itself is still readable: a primer
anchored in unique sequence before the repeat reads straight through it.
So a repeat makes the designer step back rather than give up, and the
repeat is reported with the position of its other copy.  pVT50 carries an
unannotated 306 bp exact inverted repeat, which is what forced this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import pcr, thermo
from .design import Issue
from .oligo import Oligo
from .seqio import Record, gc_fraction, revcomp


def construct_of(design) -> tuple[Record, tuple[int, int]]:
    """A finished Design as a circular molecule, and where its insert sits.

    `Design.construct` is the backbone followed by the digested insert, so
    the insert occupies the tail of the string and the two junctions fall
    at `len(backbone)` and at the origin.  The ligated product is circular
    and the panel has to be able to read across the origin, so it is
    handed on as a circular Record rather than as a string.
    """
    n = len(design.construct)
    backbone = len(design.backbone)
    record = Record(
        f"{design.insert.name}_in_{design.vector.name}",
        design.construct,
        circular=True,
    )
    return record, (backbone, (backbone + len(design.insert_fragment)) % n)


# ---------------------------------------------------------------------------
# what a Sanger read actually gives you
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadGeometry:
    """How much of a trace is usable, measured from the primer's 3' base.

    `dead_zone` is the run of bases immediately after the primer that
    cannot be called reliably; `usable` is the last base that can.  The
    numbers are deliberately conservative capillary-Sanger figures: a read
    that turns out better costs nothing, whereas a panel planned on an
    optimistic read length leaves a gap that is only discovered after
    paying for the reactions.  Facilities differ and a difficult template
    reads shorter, so these are meant to be replaced when you know better.
    """

    dead_zone: int = 50
    usable: int = 700
    overlap: int = 100
    lead: int = 50  # margin so a read starts before what it must cover

    def forward_read(self, three_prime: int) -> tuple[int, int]:
        """Reliably-read span for a forward primer ending at `three_prime`."""
        return three_prime + self.dead_zone + 1, three_prime + self.usable

    def reverse_read(self, three_prime: int) -> tuple[int, int]:
        return three_prime - self.usable, three_prime - self.dead_zone - 1


@dataclass(frozen=True)
class PrimerSpec:
    """What counts as an acceptable plain (untailed) primer."""

    target_tm: float = 58.0
    min_length: int = 18
    max_length: int = 25
    # How far from the nominal position the search may wander to find a
    # better primer: wide enough to escape an AT-rich patch, narrow enough
    # that the tiling stays on plan.
    search_window: int = 80
    # A second priming site only matters if it would prime under cycle
    # sequencing.  A perfect 3' match of this length is a real risk;
    # shorter is not.  This is also the word length of the repeat index,
    # which is what makes the two consistent.
    min_unique_anchor: int = 12
    conditions: thermo.Conditions = thermo.DEFAULT_CONDITIONS


# ---------------------------------------------------------------------------
# universal primers
# ---------------------------------------------------------------------------

# Priming sites a sequencing facility already stocks.  Finding one in the
# construct saves ordering an oligo, so the panel looks here first.
#
# These are matched against the construct by sequence, so a site missing
# from this table, or whose sequence differs between suppliers, simply is
# not found and the panel designs a custom primer instead, which works.
# Absence here means "not in this table", never "not available" -- check
# the vector's own map for the primers its supplier lists.
UNIVERSAL_PRIMERS: list[tuple[str, str]] = [
    ("M13F(-20)", "GTAAAACGACGGCCAGT"),
    ("M13R", "CAGGAAACAGCTATGAC"),
    ("T7", "TAATACGACTCACTATAGGG"),
    ("T7term", "GCTAGTTATTGCTCAGCGG"),
    ("SP6", "ATTTAGGTGACACTATAG"),
    ("CMV-F", "CGCAAATGGGCGGTAGGCGTG"),
    ("BGH-R", "TAGAAGGCACAGTCGAGG"),
    ("hU6-F", "GAGGGCCTATTTCCCATGATT"),
    ("EGFP-C-F", "CATGGTCCTGCTGGAGTTCGTG"),
    ("EGFP-N-R", "CGTCGCCGTCCAGCTCGACCAG"),
    ("pJET1.2-F", "CGACTCACTATAGGGAGAGCGGC"),
    ("pJET1.2-R", "AAGAACATCGATTTTCCATGGCAG"),
    ("AOX1-F", "GACTGGTTCCAATTGACAAGC"),
    ("AOX1-R", "GCAAATGGCATTCTGACATCC"),
]


# ---------------------------------------------------------------------------
# where a primer can bind: an index over the construct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Repeat:
    """Two copies of the same sequence, in the same or opposite senses."""

    length: int
    first: int  # 0-based start of the copy containing the query
    second: int  # 0-based start of the other copy
    inverted: bool

    def describe(self, n: int) -> str:
        kind = "inverted" if self.inverted else "direct"
        return (
            f"{self.length} bp exact {kind} repeat: "
            f"{self.first % n + 1}-{(self.first + self.length - 1) % n + 1} "
            f"and {self.second % n + 1}-"
            f"{(self.second + self.length - 1) % n + 1}"
        )


class ConstructIndex:
    """Every k-mer of a construct, on both strands.

    Priming is decided by the primer's 3'-terminal `k` bases -- that is
    what the simulator seeds on, and what has to be paired before a
    polymerase will extend.  So a primer is uniquely placed exactly when
    the k-mer ending at its 3' base occurs once in the molecule counting
    both strands, which this makes an O(1) question instead of a search
    per candidate.  It is the same criterion `pcr.find_binding_sites`
    applies, reached a cheaper way, and the chosen primer is still put
    through the real thing once.
    """

    def __init__(self, record: Record, k: int):
        self.record = record
        self.k = k
        self.n = len(record.seq)
        seq = record.seq
        haystack = seq + seq[: k - 1] if record.circular else seq
        positions: dict[str, list[int]] = {}
        for i in range(len(haystack) - k + 1):
            positions.setdefault(haystack[i : i + k], []).append(i)
        self._positions = positions

    def occurrences(self, kmer: str) -> list[tuple[int, int]]:
        """(start, sense) for every place this k-mer sits, both strands."""
        out = [(p, 1) for p in self._positions.get(kmer, ())]
        out += [(p, -1) for p in self._positions.get(revcomp(kmer), ())]
        return out

    def is_unique(self, kmer: str) -> bool:
        return len(self.occurrences(kmer)) == 1

    def repeat_at(self, start: int) -> Repeat | None:
        """Full extent of the repeat containing the k-mer at `start`.

        The k-mer only proves the repeat exists; extending the match in
        both directions is what turns it into something worth reporting,
        because "a 12-mer occurs twice" is noise and "306 bp occurs twice
        in opposite senses" is the reason the primer had to move.
        """
        seq = self.record.seq
        n = self.n
        kmer = (seq + seq[: self.k])[start % n : start % n + self.k]
        if len(kmer) < self.k:
            return None
        others = [
            (p % n, s)
            for p, s in self.occurrences(kmer)
            if not (s == 1 and p % n == start % n)
        ]
        if not others:
            return None

        def base(i: int) -> str:
            return seq[i % n]

        comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
        best: Repeat | None = None
        for other, sense in others:
            a, b, length = start % n, other, self.k
            if sense == 1:
                while length < n and base(a + length) == base(b + length):
                    length += 1
                while length < n and base(a - 1) == base(b - 1):
                    a, b, length = a - 1, b - 1, length + 1
            else:
                # The other copy is reverse complemented, so extending one
                # copy to the right extends the other to the left.
                while length < n and base(a + length) == comp.get(
                    base(b - 1), ""
                ):
                    b, length = b - 1, length + 1
                while length < n and base(a - 1) == comp.get(
                    base(b + length), ""
                ):
                    a, length = a - 1, length + 1
            found = Repeat(length, a % n, b % n, sense == -1)
            if best is None or found.length > best.length:
                best = found
        return best


# ---------------------------------------------------------------------------
# a designed or discovered verification primer
# ---------------------------------------------------------------------------


@dataclass
class VerifyPrimer:
    """One sequencing or colony-PCR primer, in construct coordinates.

    `three_prime` is a *linear* coordinate in the planning frame, which
    may run past the end of a circular construct or below zero; that is
    what makes the coverage arithmetic tractable.  `site` is the same
    position wrapped back onto the molecule, which is what gets printed.
    """

    name: str
    sequence: str
    strand: int  # +1 reads towards higher coordinates
    three_prime: int
    length_of_construct: int
    tm: float
    universal: str = ""
    drift: int = 0  # how far it had to move from where it was wanted

    @property
    def site(self) -> int:
        return self.three_prime % self.length_of_construct

    @property
    def is_universal(self) -> bool:
        return bool(self.universal)

    def read(self, geom: ReadGeometry) -> tuple[int, int]:
        if self.strand > 0:
            return geom.forward_read(self.three_prime)
        return geom.reverse_read(self.three_prime)

    def describe(self, geom: ReadGeometry) -> str:
        arrow = "-->" if self.strand > 0 else "<--"
        lo, hi = self.read(geom)
        n = self.length_of_construct
        tag = f"  [{self.universal}]" if self.universal else ""
        return (
            f"{self.name:<10} {arrow} 3' end {self.site + 1:>6}  "
            f"{len(self.sequence):2} nt  Tm {self.tm:4.1f}  "
            f"reads {lo % n + 1:>6}-{hi % n + 1:<6}{tag}"
        )


# ---------------------------------------------------------------------------
# picking a plain primer at a given place
# ---------------------------------------------------------------------------


def _slice(record: Record, start: int, end: int) -> str:
    """Linear window onto a circular molecule, wrapping as needed."""
    n = len(record.seq)
    if end - start >= n:
        raise ValueError("window longer than the molecule")
    if end <= start:
        return ""
    if not record.circular:
        if start < 0 or end > n:
            return ""
        return record.seq[start:end]
    return record.window(start % n, end % n)


def _base_score(seq: str, spec: PrimerSpec, drift: int) -> float:
    """Lower is better.  A plain primer, so there is no tail to discount.

    Sequencing and colony-PCR primers are judged more loosely on melting
    temperature than cloning primers -- a single pass, not a matched pair
    driving an amplification whose yield has to be good -- and strictly on
    having no second priming site, which is settled before scoring.

    Secondary structure is deliberately left out here and added by
    `_score` for the finalists only.  Folding is by far the most
    expensive term and the search covers thousands of candidates per
    position once a repeat forces a wide backoff, so paying for it on all
    of them would dominate the runtime to no purpose: a primer with a
    hopeless melting temperature is not rescued by folding nicely.
    """
    tm = thermo.tm(seq, spec.conditions)
    gc = gc_fraction(seq)
    penalty = abs(tm - spec.target_tm)
    if not 0.40 <= gc <= 0.60:
        penalty += 30 * min(abs(gc - 0.40), abs(gc - 0.60))
    if thermo.longest_run(seq) >= 4:
        penalty += 6
    clamp = thermo.gc_clamp(seq)
    if clamp == 0:
        penalty += 5
    elif clamp >= 4:
        penalty += 3
    if seq[-1] not in "GC":
        penalty += 1.5
    if thermo.three_prime_dg(seq) < -6.0:
        penalty += 3
    # Moving away from the nominal position is allowed but not free: it
    # spends the read budget the tiling was planned with.
    penalty += 0.08 * abs(drift)
    return penalty


def _score(seq: str, spec: PrimerSpec, drift: int) -> float:
    """`_base_score` plus the folding burden of the whole oligo."""
    return (
        _base_score(seq, spec, drift)
        + Oligo(name="x", anneal=seq, conditions=spec.conditions).structure_penalty
    )


def _candidate(
    construct: Record, three_prime: int, length: int, strand: int
) -> str | None:
    """The primer whose 3' base sits at `three_prime`, or None if unusable."""
    if strand > 0:
        seq = _slice(construct, three_prime - length + 1, three_prime + 1)
    else:
        seq = revcomp(_slice(construct, three_prime, three_prime + length))
    if len(seq) != length or set(seq) - set("ACGT"):
        return None
    return seq


@dataclass
class Placement:
    """The outcome of trying to put a primer somewhere."""

    primer: VerifyPrimer | None
    repeat: Repeat | None = None
    note: str = ""


def pick_primer(
    construct: Record,
    nominal: int,
    strand: int,
    *,
    name: str,
    spec: PrimerSpec = PrimerSpec(),
    index: ConstructIndex | None = None,
    backoff: int = 0,
) -> Placement:
    """Best plain primer whose 3' end sits at or behind `nominal`.

    The search is over both position and length, because moving a primer
    one base can turn a run of four A's at its 3' end into a G.  Positions
    whose 3'-terminal k-mer is not unique are discarded before scoring, so
    a repeat costs nothing to step over.

    `backoff` extends the search upstream *in the reading direction* --
    lower coordinates for a forward primer, higher for a reverse one.
    That is the direction that keeps a sequencing read covering what it
    was meant to cover, and that grows rather than shrinks a colony-PCR
    product, so it is safe for both uses.
    """
    index = index or ConstructIndex(construct, spec.min_unique_anchor)
    n = len(construct.seq)
    k = spec.min_unique_anchor

    scored: list[tuple[float, int, str]] = []
    blocked: Repeat | None = None
    for frame_drift in range(-(spec.search_window + backoff), spec.search_window + 1):
        pos = nominal + frame_drift * strand
        anchor_start = (pos - k + 1) if strand > 0 else pos
        kmer = _slice(construct, anchor_start, anchor_start + k)
        if len(kmer) != k or set(kmer) - set("ACGT"):
            continue
        if not index.is_unique(kmer):
            if blocked is None and abs(frame_drift) <= spec.search_window:
                blocked = index.repeat_at(anchor_start)
            continue
        for length in range(spec.min_length, spec.max_length + 1):
            seq = _candidate(construct, pos, length, strand)
            if seq is None:
                continue
            try:
                scored.append((_base_score(seq, spec, frame_drift), pos, seq))
            except thermo.ThermoError:
                continue

    # Cheap screen first, then rescore the survivors with the folding term.
    scored.sort(key=lambda row: row[0])
    finalists = sorted(
        (
            (_score(seq, spec, (pos - nominal) * strand), pos, seq)
            for _, pos, seq in scored[:60]
        ),
        key=lambda row: row[0],
    )
    for _, pos, seq in finalists[:12]:
        # The index settles the 3' anchor; confirm the whole oligo against
        # the same model the simulator uses before committing to it.
        sites = pcr.find_binding_sites(
            seq, construct, min_anchor=k, conditions=spec.conditions
        )
        if len(sites) != 1:
            continue
        drift = (pos - nominal) * strand
        return Placement(
            primer=VerifyPrimer(
                name=name,
                sequence=seq,
                strand=strand,
                three_prime=pos,
                length_of_construct=n,
                tm=thermo.tm(seq, spec.conditions),
                drift=drift,
            ),
            repeat=blocked if drift < -spec.search_window else None,
        )
    return Placement(primer=None, repeat=blocked)


def find_universal(
    construct: Record, *, spec: PrimerSpec = PrimerSpec()
) -> list[VerifyPrimer]:
    """Universal priming sites present in the finished construct.

    The search is on the construct rather than the parental vector on
    purpose.  A site inside a cassette the cloning replaces is gone from
    the product: EGFP primers are listed on the map of an EGFP vector, but
    once EGFP has been swapped out they prime on nothing.
    """
    n = len(construct.seq)
    found: list[VerifyPrimer] = []
    for label, seq in UNIVERSAL_PRIMERS:
        sites = [
            s
            for s in pcr.find_binding_sites(
                seq, construct, min_anchor=15, conditions=spec.conditions
            )
            if s.mismatches == 0 and s.anchor >= len(seq) - 2
        ]
        if len(sites) != 1:
            continue  # absent, or ambiguous and therefore useless
        site = sites[0]
        found.append(
            VerifyPrimer(
                name=label,
                sequence=seq,
                strand=site.strand,
                three_prime=site.three_prime,
                length_of_construct=n,
                tm=thermo.tm(seq, spec.conditions),
                universal=label,
            )
        )
    return found


# ---------------------------------------------------------------------------
# sequencing panel
# ---------------------------------------------------------------------------


@dataclass
class SequencingPlan:
    construct: Record
    geometry: ReadGeometry
    target: tuple[int, int]  # the span that must be verified
    region: tuple[int, int] = (0, 0)  # target plus junction flanks
    forward: list[VerifyPrimer] = field(default_factory=list)
    reverse: list[VerifyPrimer] = field(default_factory=list)
    universal_available: list[VerifyPrimer] = field(default_factory=list)
    repeats: list[Repeat] = field(default_factory=list)
    gaps: list[tuple[int, int]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def primers(self) -> list[VerifyPrimer]:
        return self.forward + self.reverse

    @property
    def to_order(self) -> list[VerifyPrimer]:
        return [p for p in self.primers if not p.is_universal]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _tile(
    construct: Record,
    region: tuple[int, int],
    strand: int,
    geom: ReadGeometry,
    spec: PrimerSpec,
    index: ConstructIndex,
    universals: list[VerifyPrimer],
    prefix: str,
) -> tuple[list[VerifyPrimer], list[Repeat], list[Issue]]:
    """Walk one strand across `region`, reusing universal primers first."""
    n = len(construct.seq)
    start, end = region
    chosen: list[VerifyPrimer] = []
    repeats: list[Repeat] = []
    issues: list[Issue] = []

    # The walk runs along the strand being read, so the reverse pass is
    # the same arithmetic with the coordinates mirrored.
    def frame(x: int) -> int:
        return x if strand > 0 else -x

    cursor = frame(start) if strand > 0 else frame(end)
    stop = frame(end) if strand > 0 else frame(start)

    # A universal primer's linear position on a circular construct is only
    # defined up to a whole turn; take the turn that puts its read nearest
    # the region being covered.
    pending: list[tuple[int, int, VerifyPrimer]] = []
    midpoint = (frame(start) + frame(end)) / 2
    for u in universals:
        if u.strand != strand:
            continue
        turns = round((midpoint - frame(u.three_prime)) / n)
        shifted = VerifyPrimer(
            name=u.name,
            sequence=u.sequence,
            strand=u.strand,
            three_prime=u.three_prime + turns * n * strand,
            length_of_construct=n,
            tm=u.tm,
            universal=u.universal,
        )
        lo, hi = shifted.read(geom)
        pending.append((frame(lo) if strand > 0 else frame(hi),
                        frame(hi) if strand > 0 else frame(lo), shifted))

    index_number = 1
    guard = 0
    while cursor < stop:
        guard += 1
        if guard > 64:
            issues.append(Issue("error", f"{prefix} tiling did not terminate"))
            break

        # Is a universal primer already reading from here, and far enough
        # to be worth using instead of an oligo that has to be ordered?
        best = None
        for lo, hi, primer in pending:
            if lo <= cursor and hi > cursor + geom.overlap:
                if best is None or hi > best[1]:
                    best = (lo, hi, primer)
        if best is not None:
            _, hi, primer = best
            chosen.append(primer)
            pending = [row for row in pending if row[2] is not primer]
            cursor = hi - geom.overlap
            continue

        nominal = frame(cursor - geom.dead_zone - 1 - geom.lead)
        # How far back the primer may go and still leave a read that
        # advances past the cursor by at least the required overlap.
        room = geom.usable - geom.dead_zone - geom.lead - 2 * geom.overlap
        placed = pick_primer(
            construct, nominal, strand,
            name=f"{prefix}{index_number}", spec=spec, index=index,
            backoff=max(room, 0),
        )
        if placed.repeat is not None:
            repeats.append(placed.repeat)
        if placed.primer is None:
            reach = cursor + geom.usable - geom.dead_zone - geom.overlap
            issues.append(
                Issue(
                    "error",
                    f"no uniquely-binding "
                    f"{'forward' if strand > 0 else 'reverse'} primer within "
                    f"reach of position {frame(cursor) % n + 1}"
                    + (
                        f"; {placed.repeat.describe(n)}"
                        if placed.repeat
                        else ""
                    ),
                )
            )
            cursor = reach
            continue

        lo, hi = placed.primer.read(geom)
        reach = frame(hi) if strand > 0 else frame(lo)
        if reach <= cursor:
            issues.append(Issue("error", f"{prefix} tiling stalled"))
            break
        chosen.append(placed.primer)
        index_number += 1
        cursor = reach - geom.overlap

    return chosen, repeats, issues


def _coverage_gaps(
    primers: list[VerifyPrimer], region: tuple[int, int], geom: ReadGeometry
) -> list[tuple[int, int]]:
    """Parts of `region` no read reaches, in the planning frame."""
    spans = sorted(p.read(geom) for p in primers)
    start, end = region
    gaps: list[tuple[int, int]] = []
    cursor = start
    for lo, hi in spans:
        if lo > cursor:
            gaps.append((cursor, min(lo, end)))
        cursor = max(cursor, hi + 1)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return [g for g in gaps if g[1] > g[0]]


def sequencing_plan(
    construct: Record,
    target: tuple[int, int],
    *,
    geometry: ReadGeometry = ReadGeometry(),
    spec: PrimerSpec = PrimerSpec(),
    junction_flank: int = 60,
    strands: str = "both",
) -> SequencingPlan:
    """Primers that read the whole of `target` plus both of its junctions.

    `target` is the span that has to be verified -- normally the insert
    fragment -- in construct coordinates.  It is extended by
    `junction_flank` at each end so the reads carry into vector sequence
    either side, which is what makes a junction readable: a trace that
    stops at the join proves nothing about the join.
    """
    n = len(construct.seq)
    t_start, t_end = target
    span = (t_end - t_start) % n or n
    region = (t_start - junction_flank, t_start + span + junction_flank)

    plan = SequencingPlan(
        construct=construct, geometry=geometry, target=target, region=region
    )
    index = ConstructIndex(construct, spec.min_unique_anchor)
    plan.universal_available = find_universal(construct, spec=spec)

    if region[1] - region[0] >= n:
        plan.issues.append(
            Issue(
                "warning",
                "the region to verify is effectively the whole construct, so "
                "the panel reads it as a circle and the flanks overlap",
            )
        )
        region = (region[0], region[0] + n - 1)

    fwd, frepeats, fissues = _tile(
        construct, region, +1, geometry, spec, index,
        plan.universal_available, "seqF",
    )
    plan.forward = fwd
    plan.issues.extend(fissues)
    plan.repeats.extend(frepeats)

    if strands == "both":
        used = {p.name for p in fwd if p.is_universal}
        rev, rrepeats, rissues = _tile(
            construct, region, -1, geometry, spec, index,
            [u for u in plan.universal_available if u.name not in used],
            "seqR",
        )
        plan.reverse = rev
        plan.issues.extend(rissues)
        plan.repeats.extend(rrepeats)

    plan.gaps = _coverage_gaps(plan.forward, region, geometry)
    if strands == "both":
        plan.gaps = sorted(
            set(plan.gaps) | set(_coverage_gaps(plan.reverse, region, geometry))
        )
    for lo, hi in plan.gaps:
        plan.issues.append(
            Issue(
                "error",
                f"{hi - lo} bp at {lo % n + 1}-{(hi - 1) % n + 1} is not "
                "covered by any read",
            )
        )

    # Report each distinct repeat once, with where the other copy sits.
    seen: set[tuple[int, int, int, bool]] = set()
    for r in plan.repeats:
        key = tuple(sorted((r.first, r.second))) + (r.length, r.inverted)
        if key in seen:
            continue
        seen.add(key)
        plan.issues.append(
            Issue(
                "warning",
                r.describe(n)
                + ". No primer can be anchored inside it, so the reads "
                "covering that stretch start before it and run through",
            )
        )

    # A tandem repeat is worse than an ordinary one, and worse in a way
    # that coverage does not show.  Two copies lying closer together
    # than they are long means the sequence is an array, and a Sanger
    # read running through an array cannot say how many copies it
    # passed: every repeat gives the same trace.  So even a region with
    # no gaps in it is not verified, and saying "contiguous coverage"
    # without saying this would be the misleading half of a true
    # statement.
    tandem = [
        r
        for r in plan.repeats
        if not r.inverted
        and 0 < abs(r.first - r.second) % n <= r.length
    ]
    if tandem:
        worst = max(tandem, key=lambda r: r.length)
        period = abs(worst.first - worst.second) % n
        plan.issues.append(
            Issue(
                "warning",
                f"the insert contains a tandem array with a period of about "
                f"{period} bp. Sanger reads can run through it but cannot "
                "count the copies, because each repeat gives the same "
                "trace, so coverage of that stretch is not verification of "
                "it. Count the copies another way: a diagnostic digest "
                "sized on a gel, or a long read that spans the whole array",
            )
        )

    moved = [p for p in plan.primers if abs(p.drift) > spec.search_window]
    if moved:
        plan.issues.append(
            Issue(
                "note",
                f"{len(moved)} primer(s) sit further back than planned to "
                "clear a repeat, which spends read length but keeps the "
                "coverage contiguous",
            )
        )

    if strands == "both":
        plan.issues.append(
            Issue(
                "note",
                f"both strands are covered: {len(plan.forward)} forward and "
                f"{len(plan.reverse)} reverse reads. A routine subclone is "
                "often passed on single-pass coverage, which halves the "
                "cost; read both strands for anything you will build on, "
                "because one trace cannot distinguish a real variant from a "
                "sequencing artefact",
            )
        )
    else:
        plan.issues.append(
            Issue(
                "warning",
                "single-pass coverage only. An ambiguous base in one trace "
                "has nothing to be checked against; read both strands "
                "before trusting a difference from the expected sequence",
            )
        )

    reused = [p for p in plan.primers if p.is_universal]
    if reused:
        plan.issues.append(
            Issue(
                "note",
                f"{len(reused)} read(s) use a universal primer the facility "
                "already stocks: " + ", ".join(p.universal for p in reused),
            )
        )
    elif plan.universal_available:
        plan.issues.append(
            Issue(
                "note",
                "universal sites exist in the construct ("
                + ", ".join(u.universal for u in plan.universal_available)
                + ") but none reads into the region to verify",
            )
        )
    else:
        plan.issues.append(
            Issue(
                "note",
                "no universal priming site from the built-in table was found "
                "in the construct, so every read needs a custom primer; "
                "check the vector's own map, which may list primers this "
                "table does not carry",
            )
        )

    plan.issues.append(
        Issue(
            "note",
            f"primers sit at least {geometry.dead_zone} bases back from what "
            "they read, because the bases just after a primer are not called "
            "reliably; a primer placed on a junction reads everything except "
            "the junction",
        )
    )
    return plan


# ---------------------------------------------------------------------------
# colony PCR
# ---------------------------------------------------------------------------


@dataclass
class ColonyAssay:
    """One primer pair and what its result means."""

    name: str
    question: str
    forward: VerifyPrimer
    reverse: VerifyPrimer
    recombinant_bp: int | None
    parental_bp: int | None
    parental_checked: bool
    spans_junction: bool
    issues: list[Issue] = field(default_factory=list)

    @property
    def discriminates(self) -> bool:
        """Does a band actually distinguish a clone from the background?"""
        if self.recombinant_bp is None:
            return False
        if self.parental_bp is None:
            return True
        return abs(self.recombinant_bp - self.parental_bp) > 0.2 * max(
            self.recombinant_bp, self.parental_bp
        )


@dataclass
class ColonyPlan:
    construct: Record
    polymerase: object
    assays: list[ColonyAssay] = field(default_factory=list)
    repeats: list[Repeat] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"] + [
            i for a in self.assays for i in a.issues if i.level == "error"
        ]

    @property
    def usable(self) -> list[ColonyAssay]:
        return [
            a
            for a in self.assays
            if a.recombinant_bp is not None and a.discriminates
        ]

    @property
    def ok(self) -> bool:
        return bool(self.usable)


def _product_length(
    primers: dict[str, str], template: Record, polymerase, spec: PrimerSpec
) -> int | None:
    """Length of the cleanest product these two primers give on `template`."""
    sim = pcr.simulate(
        primers,
        [template],
        polymerase=polymerase,
        annealing_temp=55.0,
        conditions=spec.conditions,
        min_anchor=spec.min_unique_anchor,
    )
    clean = [p for p in sim.products if p.is_clean]
    if not clean:
        return None
    return min(p.length for p in clean)


def colony_pcr_plan(
    construct: Record,
    target: tuple[int, int],
    *,
    parental: Record | None = None,
    polymerase=None,
    spec: PrimerSpec = PrimerSpec(),
    anchor_distance: int = 250,
    max_screen_bp: int = 2000,
) -> ColonyPlan:
    """Primer pairs that tell a real clone from the background.

    `target` is the insert's span in the construct; everything outside it
    is backbone.  `parental` is the uncut destination vector -- the
    commonest false positive -- and is screened against directly rather
    than assumed away.
    """
    polymerase = polymerase or pcr.POLYMERASES["taq"]
    n = len(construct.seq)
    t_start, t_end = target
    span = (t_end - t_start) % n or n
    plan = ColonyPlan(construct=construct, polymerase=polymerase)
    index = ConstructIndex(construct, spec.min_unique_anchor)

    backbone = n - span
    if backbone < 2 * anchor_distance + spec.max_length:
        plan.issues.append(
            Issue(
                "error",
                f"only {backbone} bp of backbone outside the insert, which is "
                "not enough room to anchor a primer in the vector",
            )
        )
        return plan

    # Room to step away from the junction when a repeat is in the way.
    # Every one of these directions runs away from the junction, so a
    # backed-off primer only ever lengthens the product.
    room = max_screen_bp // 2

    def place(name, nominal, strand, limit):
        placed = pick_primer(
            construct, nominal, strand, name=name, spec=spec, index=index,
            backoff=limit,
        )
        if placed.repeat is not None:
            plan.repeats.append(placed.repeat)
        return placed.primer

    vec_f = place("colVecF", t_start - anchor_distance, +1, room)
    vec_r = place("colVecR", t_start + span + anchor_distance, -1, room)
    ins_r = place("colInsR", t_start + anchor_distance, -1,
                  max(min(room, span - anchor_distance - spec.max_length), 0))
    ins_f = place("colInsF", t_start + span - anchor_distance, +1,
                  max(min(room, span - anchor_distance - spec.max_length), 0))

    missing = [
        label
        for label, p in (
            ("backbone 5'", vec_f),
            ("backbone 3'", vec_r),
            ("insert 5'", ins_r),
            ("insert 3'", ins_f),
        )
        if p is None
    ]
    if missing:
        plan.issues.append(
            Issue(
                "warning",
                "no uniquely-binding primer could be placed in: "
                + ", ".join(missing),
            )
        )

    def add(name, question, fwd, rev, spans):
        if fwd is None or rev is None:
            return
        primers = {fwd.name: fwd.sequence, rev.name: rev.sequence}
        rec = _product_length(primers, construct, polymerase, spec)
        par = (
            _product_length(primers, parental, polymerase, spec)
            if parental is not None
            else None
        )
        assay = ColonyAssay(
            name=name, question=question, forward=fwd, reverse=rev,
            recombinant_bp=rec, parental_bp=par,
            parental_checked=parental is not None, spans_junction=spans,
        )
        if rec is None:
            assay.issues.append(
                Issue("error", "no product on the construct; pair unusable")
            )
        elif rec > max_screen_bp:
            assay.issues.append(
                Issue(
                    "warning",
                    f"{rec} bp is long for a colony screen on crude lysate; "
                    "expect weak or inconsistent bands",
                )
            )
        if not assay.parental_checked:
            assay.issues.append(
                Issue(
                    "warning",
                    "the parental vector was not supplied, so this pair was "
                    "not screened against the commonest false positive",
                )
            )
        elif par is None:
            assay.issues.append(
                Issue(
                    "note",
                    "the parental vector gives no product, so a band means "
                    "the insert is joined to the vector and not merely "
                    "present in the colony",
                )
            )
        elif not assay.discriminates:
            assay.issues.append(
                Issue(
                    "error",
                    f"the parental vector gives {par} bp, too close to the "
                    f"{rec} bp recombinant band to call apart on a gel",
                )
            )
        else:
            assay.issues.append(
                Issue(
                    "note",
                    f"uncut or religated parental vector gives {par} bp, "
                    "which is the band a negative colony shows",
                )
            )
        plan.assays.append(assay)

    add(
        "5' junction",
        "is the insert joined to the vector at its 5' end, the right way "
        "round?",
        vec_f, ins_r, True,
    )
    add(
        "3' junction",
        "is the insert joined to the vector at its 3' end?",
        ins_f, vec_r, True,
    )
    if vec_f is not None and vec_r is not None:
        flanking_bp = span + (t_start - vec_f.three_prime) + (
            vec_r.three_prime - t_start - span
        )
        if flanking_bp <= max_screen_bp:
            add("flanking", "empty or full, by size, in one reaction",
                vec_f, vec_r, False)
        else:
            plan.issues.append(
                Issue(
                    "note",
                    f"a flanking vector pair would give about {flanking_bp} bp "
                    "across this insert, too long to screen reliably from a "
                    "colony; the junction pairs are the usable assay here",
                )
            )

    seen: set[tuple] = set()
    for r in plan.repeats:
        key = tuple(sorted((r.first, r.second))) + (r.length, r.inverted)
        if key in seen:
            continue
        seen.add(key)
        plan.issues.append(
            Issue(
                "warning",
                r.describe(n)
                + ". A primer there would bind both copies, so the pair was "
                "anchored outside it and the product is correspondingly "
                "longer",
            )
        )

    # The point of the whole arrangement.
    plan.issues.append(
        Issue(
            "note",
            "each junction pair has one primer in the vector backbone and one "
            "in the insert. A pair with both primers inside the insert would "
            "amplify leftover template and PCR product carried through the "
            "ligation just as well as a real recombinant, so every colony "
            "would score positive",
        )
    )
    if any(a.spans_junction for a in plan.assays):
        plan.issues.append(
            Issue(
                "note",
                "a junction pair also reports orientation: an insert ligated "
                "backwards gives no product at all, whereas a flanking vector "
                "pair gives the same size either way",
            )
        )
    plan.issues.append(
        Issue(
            "warning",
            "colony PCR shows that a junction exists, not that the sequence is "
            "right. It cannot see a point mutation introduced by the cloning "
            "PCR, so a positive colony still has to be sequenced",
        )
    )
    plan.issues.append(
        Issue(
            "note",
            f"{polymerase.name} is the usual choice here: fidelity does not "
            "matter for a screen, and tolerance of crude lysate does. Lyse the "
            "colony first, or the early cycles go on releasing template",
        )
    )
    return plan
