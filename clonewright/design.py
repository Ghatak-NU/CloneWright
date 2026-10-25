"""Restriction-cloning primer design.

The design is built from the vector outwards.  An enzyme pair is only
admissible if it cuts the vector exactly once each, never cuts the insert,
and leaves ends that cannot religate on themselves.  Only then are
annealing regions chosen, and their melting temperatures are matched to
each other rather than to a nominal target in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import enzymes as ez
from . import thermo
from .oligo import Duplex, Oligo, worst_duplex
from .seqio import Feature, Record, gc_fraction, revcomp

STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# Translation initiation contexts.  The mammalian Kozak is GCCACC; yeasts
# and other fungi instead prefer an A-rich stretch before the start codon,
# so putting a mammalian Kozak in front of a gene going into a yeast vector
# is not merely useless, it is the wrong sequence for the host.  Both keep
# a purine at -3, the position that matters most in either system.
KOZAK_CONTEXTS: dict[str, tuple[str, str]] = {
    "mammalian": ("GCCACC", "mammalian Kozak"),
    "yeast": ("AAAACA", "yeast/fungal A-rich context"),
    # Bacteria have no Kozak.  Initiation is driven by a Shine-Dalgarno
    # sequence sitting a short distance upstream of the start codon, and
    # in an expression vector that sequence belongs to the *vector*, not
    # to your primer: adding bases here would only push the ribosome
    # binding site further from the ATG and make initiation worse.  So
    # the entry is empty on purpose, and the real check -- that the
    # vector's RBS ends up the right distance away -- is `rbs_spacing`.
    "bacterial": ("", "no added context; the vector's RBS drives initiation"),
    "none": ("", "no added context"),
}

# Shine-Dalgarno consensus and the window it has to sit in.  The spacing
# is measured from the last base of the RBS to the A of the ATG; too
# close and the ribosome cannot reach the start codon, too far and it
# initiates poorly or not at all.  The usable range is wider than the
# optimum, so the two are kept apart.
SHINE_DALGARNO = "AGGAGG"
RBS_SPACING_OPTIMAL = (5, 13)
RBS_SPACING_USABLE = (3, 20)

# Feature names that betray which host a vector expresses in.
_MAMMALIAN_HINTS = (
    "cmv", "ef-1", "ef1a", "ef1alpha", "sv40 promoter", "pgk promoter",
    "cag", "ubc", "rsv promoter", "cba",
)
_FUNGAL_HINTS = (
    "adh1", "tef1", "tef promoter", "gpd", "tdh3", "pgk1", "cyc1", "gal1",
    "act1", "ura3", "leu2", "his3", "trp1", "cen/ars", "2-micron", "2u ori",
    # Integration loci and organism names seen on fungal vector maps.
    "candida", "albicans", "cerevisiae", "rp10", "rps1", "neurospora",
    "aspergillus", "pichia", "aox1",
)


# Selection markers are weaker evidence than promoters -- several work in
# more than one kingdom -- so they are only consulted when no promoter
# settles the question.
_MAMMALIAN_MARKERS = ("puror", "puromycin", "hygror", "hygromycin",
                      "blasticidin", "bsd")
_FUNGAL_MARKERS = ("ura3", "leu2", "his3", "trp1", "nat1", "sat1")

# Telling a bacterial *expression* cassette from the pUC backbone that
# every one of these plasmids carries so it can be grown up in E. coli is
# the whole difficulty here, and it is easy to get wrong in the direction
# that calls everything bacterial.
#
# A T7 promoter, a lac operator, a lac promoter and lacI are all present
# on mammalian and fungal vectors: they are sequencing-primer sites and
# backbone leftovers.  pLJM1 (lentiviral) and pVT50 (fungal) each carry a
# T7 promoter, a lac operator *and* a lac promoter, and neither expresses
# a thing in a bacterium.  So none of those is evidence on its own, for
# the same reason an AmpR marker is not.
#
# What does discriminate:
#   * an annotated ribosome binding site -- no eukaryotic vector has one,
#     because nothing in a eukaryote reads it
#   * a T7 promoter together with a T7 terminator -- a whole cassette
#     rather than a stray priming site
#   * a named bacterial expression backbone
_RBS_WORDS = ("shine", "dalgarno", "ribosome binding")
_BACTERIAL_BACKBONES = (
    "pet", "pgex", "pmal", "tac promoter", "trc promoter", "pbad",
    "arabad promoter",
    # cspA is E. coli's own cold-shock gene. Its promoter and 5' UTR are
    # what pCold is built from, and no eukaryotic vector annotates them.
    # pCold needs naming separately because it annotates no ribosome
    # binding site at all -- initiation runs off the cspA 5' UTR and a
    # downstream box, neither of which is marked as an RBS.
    "cspa",
)


def _has_rbs(vector: Record) -> bool:
    for f in vector.features:
        if f.kind.lower() == "rbs":
            return True
        name = f.name.lower()
        if name == "rbs" or any(w in name for w in _RBS_WORDS):
            return True
    return False


def detect_host(vector: Record) -> str | None:
    """Guess the expression host from the vector's annotations.

    Returns "bacterial", "mammalian", "yeast", or None when the
    annotations do not say.  This only ever reads feature names, so a
    vector with a bare backbone annotation yields None rather than a
    guess.

    Bacterial is tested first and on different evidence from the others.
    A eukaryotic promoter can sit on a shuttle vector that also replicates
    in E. coli, so finding a CMV promoter does not rule E. coli out -- but
    an annotated RBS or a T7/lac cassette says translation is meant to
    happen in a bacterium, which is the question being asked.  Bacterial
    resistance markers are deliberately *not* evidence here: every one of
    these plasmids carries one so it can be grown up in E. coli, and that
    says nothing about where the protein is made.
    """
    names = " ".join(f.name.lower() for f in vector.features)
    eukaryotic = any(h in names for h in _MAMMALIAN_HINTS) or any(
        h in names for h in _FUNGAL_HINTS
    )
    bacterial = (
        _has_rbs(vector)
        or ("t7 promoter" in names and "t7 terminator" in names)
        or any(h in names for h in _BACTERIAL_BACKBONES)
    )
    if bacterial and eukaryotic:
        # A dual-host construct, or an annotation that means something
        # other than what it looks like.  Either way the vector cannot
        # answer the question, and saying so beats picking a side.
        return None
    if bacterial:
        return "bacterial"
    if any(h in names for h in _MAMMALIAN_HINTS):
        return "mammalian"
    if any(h in names for h in _FUNGAL_HINTS):
        return "yeast"
    if any(h in names for h in _MAMMALIAN_MARKERS):
        return "mammalian"
    if any(h in names for h in _FUNGAL_MARKERS):
        return "yeast"
    return None


# Promoters that drive a bacterial expression cassette.  A promoter only
# counts here if it is the one transcribing the insert, which is why the
# match is on a promoter feature rather than on the name appearing
# anywhere: pET-28a's "T7 terminator" and a T7 sequencing-primer site are
# both called T7 and neither transcribes anything.
_DRIVING_PROMOTERS = (
    "t7", "tac", "trc", "t5", "arabad", "pbad", "rhap", "cspa",
)

# The same question for a eukaryotic vector, where the bacterial list
# gives a confidently wrong answer rather than no answer.  A mammalian
# vector routinely carries a T7 promoter for in-vitro transcription and
# sequencing, and nothing transcribes the insert from it in a cell:
# pcDNA3.1 has one 13 bp before its MCS and a CMV promoter 76 bp before
# that, and matching the bacterial list returned the T7 and so asked for
# a lambda-DE3 lysogen to express a mammalian construct.  pLJM1-EGFP,
# one of the two original bench cases, did the same from a T7 site 2.9 kb
# away from its cassette.
#
# Ordered by how likely each is to be driving the insert rather than a
# selection marker, which is how the choice is made when the vector
# annotates no cloning site to measure from.
_MAMMALIAN_PROMOTERS = (
    "cmv", "cag", "ef-1", "ef1", "cba", "ubc", "rsv", "msv", "sv40",
    "pgk", "tk",
)
_YEAST_PROMOTERS = (
    "gal1", "gal10", "gal7", "adh1", "adh2", "tef1", "tef", "gpd",
    "tdh3", "pgk1", "cyc1", "met25", "cup1", "pho5",
)


def _promoter_features(vector: Record, wanted: tuple[str, ...]) -> list[Feature]:
    """Promoter features whose name is one of `wanted`.

    Matched on a promoter feature rather than on the name appearing
    anywhere, because pET-28a's "T7 terminator" and a T7
    sequencing-primer site are both called T7 and neither transcribes
    anything.
    """
    out: list[Feature] = []
    for f in vector.features:
        name = f.name.lower()
        if f.kind.lower() != "promoter" and "promoter" not in name:
            continue
        if "laci" in name:
            continue  # drives the repressor, not the insert
        if any(name.startswith(p) or f"{p} promoter" in name for p in wanted):
            out.append(f)
    return out


def driving_promoter(vector: Record) -> Feature | None:
    """The promoter that transcribes the insert, if the map names one.

    Which promoters can be the answer depends on the host, because the
    same vector often carries several and only one of them transcribes
    anything in the cell the construct is going into.

    For a eukaryotic vector the choice among them is geometric: the
    cassette driver is the one immediately in front of the cloning
    site, which is what separates pcDNA3.1's CMV from the SV40
    promoter 700 bp downstream that drives its NeoR marker.  Where the
    map annotates no cloning site to measure from, the ordering of the
    name lists stands in for it.

    Bacterial vectors keep the older rule -- first match in feature
    order -- because their cassettes often read against the map, so
    "in front of the cloning site" is not a question about coordinates
    there, and pET-28a's T7 promoter sits at 367-386 with its MCS at
    157-203.
    """
    host = detect_host(vector)
    if host == "mammalian":
        wanted = _MAMMALIAN_PROMOTERS
    elif host == "yeast":
        wanted = _YEAST_PROMOTERS
    else:
        return next(iter(_promoter_features(vector, _DRIVING_PROMOTERS)), None)

    candidates = _promoter_features(vector, wanted)
    if not candidates:
        return None
    marked = annotated_polylinker(vector)
    if marked is not None:
        upstream = [f for f in candidates if f.end <= marked[0]]
        if upstream:
            return max(upstream, key=lambda f: f.end)
    order = {p: i for i, p in enumerate(wanted)}

    def rank(f: Feature) -> int:
        name = f.name.lower()
        return min(
            (i for p, i in order.items()
             if name.startswith(p) or f"{p} promoter" in name),
            default=len(wanted),
        )

    return min(candidates, key=rank)


def expression_direction(vector: Record) -> int:
    """+1 if the cassette reads along the map, -1 if it reads against it.

    pET-28a is built the second way: its T7 promoter sits at 368-386 on
    the reverse strand and transcription runs down through the lac
    operator, the RBS, the ATG, the 6xHis tag and the thrombin site to
    the MCS at 158-203.  Every piece of geometry in this module assumes
    the first way, so rather than carry a direction through all of it the
    vector is turned round once, up front, and everything downstream sees
    a cassette that reads forward.

    Only bacterial vectors are considered.  A mammalian vector routinely
    carries a reverse-strand T7 site for sequencing, and flipping a
    plasmid because of a priming site would be a bad trade.
    """
    if detect_host(vector) != "bacterial":
        return 1
    promoter = driving_promoter(vector)
    if promoter is None:
        return 1
    return 1 if promoter.forward else -1


def rbs_spacing(vector: Record, atg_position: int) -> tuple[int, Feature] | None:
    """Distance from the vector's ribosome binding site to a start codon.

    Measured from the last base of the annotated RBS to the first base of
    the ATG, in the finished construct's coordinates.  Returns None when
    the vector annotates no RBS, which is a different answer from "the
    spacing is wrong" and is reported differently.
    """
    sites = [
        f
        for f in vector.features
        if f.kind.lower() == "rbs"
        or "rbs" in f.name.lower()
        or "shine" in f.name.lower()
        or "ribosome binding" in f.name.lower()
    ]
    if not sites:
        return None
    nearest = min(sites, key=lambda f: abs(atg_position - f.end))
    return atg_position - nearest.end, nearest


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning" | "note"
    text: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.text}"


@dataclass(frozen=True)
class Options:
    target_tm: float = 62.0
    anneal_min: int = 18
    anneal_max: int = 32
    max_tm_mismatch: float = 3.0
    spacer: str = "TGACTG"
    # How hard to push against hairpins and dimers when choosing an
    # annealing region.  1.0 weighs one kcal/mol of folding against one
    # degree of Tm error; raise it to buy lower structure with Tm.
    structure_weight: float = 1.0
    # Let the designer pick among Kozak variants instead of always using
    # the consensus.  Off by default so output stays predictable.
    optimize_pad: bool = False
    conditions: thermo.Conditions = thermo.DEFAULT_CONDITIONS
    mode: str = "native"  # native | nfusion | cfusion
    add_kozak: bool = True
    # Which translation initiation context to put before the start codon:
    # "auto", "mammalian", "yeast" or "none".
    kozak: str = "auto"
    five_enzyme: str | None = None
    three_enzyme: str | None = None
    window: tuple[int, int] | None = None
    window_span: int = 2000
    # Replacing a cassette removes it, so the excised piece is large and a
    # floor is meaningful.  Inserting into an empty polylinker removes
    # almost nothing, so there is no floor at all.
    min_excision: int = 0
    # Widest span of vector that still counts as one polylinker, and the
    # most vector an insertion is allowed to remove.
    polylinker_span: int = 250
    max_insert_excision: int = 250
    # Below this, cutting one site leaves the other too near a fragment end.
    close_site_warning: int = 25
    # Two-step route: park the product in a shuttle, verify, then subclone.
    two_step: bool = False
    shuttle_record: Record | None = None
    # Name of the vector feature to replace, e.g. "EGFP".  When unset the
    # designer looks for the annotated cassette with the tightest flanking
    # single-cutters and says which one it chose.
    replace: str | None = None
    # How far beyond the replaced cassette a site may sit before it is
    # penalised for taking vector sequence with it.
    max_flank_loss: int = 400
    # Required for the fusion modes: a vector coordinate (0-based) that is
    # the first base of a codon in the tag you are fusing to.
    vector_frame_5: int | None = None
    vector_frame_3: int | None = None


@dataclass
class EnzymePair:
    five: ez.Enzyme
    three: ez.Enzyme
    pos_five: int
    pos_three: int
    excision: int
    score: float
    reasons: list[str] = field(default_factory=list)
    strategy: str = "replace"  # "replace" a cassette, or "insert" into a MCS

    @property
    def label(self) -> str:
        return f"{self.five.name}/{self.three.name}"


@dataclass
class Design:
    insert: Record
    vector: Record
    pair: EnzymePair
    forward: Oligo
    reverse: Oligo
    options: Options
    core_start: int
    core_end: int
    issues: list[Issue] = field(default_factory=list)
    alternatives: list[EnzymePair] = field(default_factory=list)
    # True when the vector was reverse-complemented so its expression
    # cassette reads forward.  Positions in this Design are in that
    # turned-round frame; `on_the_map` puts them back.
    vector_flipped: bool = False

    def on_the_map(self, position: int) -> int:
        """A position here, in the coordinates the user's map uses."""
        if not self.vector_flipped:
            return position
        return len(self.vector) - position

    # ---- derived sizes -------------------------------------------------

    @property
    def core(self) -> str:
        return self.insert.seq[self.core_start : self.core_end]

    @property
    def amplicon(self) -> str:
        return (
            self.forward.tail
            + self.core
            + revcomp(self.reverse.tail)
        )

    @property
    def insert_fragment(self) -> str:
        """The amplicon after digestion with both enzymes, top strand."""
        f, r = self.pair.five, self.pair.three
        left = self.forward.site[f.top :] + self.forward.pad
        right = revcomp(self.reverse.pad) + revcomp(self.reverse.site)[: r.top]
        return left + self.core + right

    @property
    def backbone(self) -> str:
        v = self.vector
        cut5 = self.pair.pos_five + self.pair.five.top
        cut3 = self.pair.pos_three + self.pair.three.top
        return v.window(cut3, cut5)

    @property
    def construct(self) -> str:
        return self.backbone + self.insert_fragment

    @property
    def annealing_temp(self) -> float:
        """Ta from the lower annealing-region Tm, the standard -5 C rule."""
        return min(self.forward.tm_anneal, self.reverse.tm_anneal) - 5.0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# enzyme selection
# ---------------------------------------------------------------------------


def _auto_window(positions: list[int], span: int) -> tuple[int, int]:
    """Densest cluster of unique sites: usually the polylinker."""
    pts = sorted(positions)
    if not pts:
        return 0, 0
    best_key: tuple[int, int] | None = None
    best = (pts[0], pts[-1])
    for i in range(len(pts)):
        j = i
        while j + 1 < len(pts) and pts[j + 1] - pts[i] <= span:
            j += 1
        key = (j - i + 1, -(pts[j] - pts[i]))
        if best_key is None or key > best_key:
            best_key, best = key, (pts[i], pts[j])
    return best


# How far an annotated cassette may sit from the cloning site and still
# be the thing that site is for.  Measured: 0 bp on pUC19 (lacZ-alpha
# encloses its MCS), 55 on pMAL-c5X (MBP, across the Factor Xa site),
# against 324 bp for the nearest backbone gene in the set.
CASSETTE_REACH = 150


def _fusion_leader_features(vector: Record) -> tuple[Feature, ...]:
    """The coding sequence an N-terminal fusion is built behind.

    Whatever the insert fuses to has to survive the excision, which is
    the same requirement `downstream_partner` carries on the 3' side.
    """
    frame = fusion_frame(vector)
    marked = annotated_polylinker(vector)
    if frame is None or not frame.is_confirmed or marked is None:
        return ()
    return tuple(f for f in frame.confirmed_by if f.start < marked[0])


def _auto_cassette(
    vector: Record,
    cuts: dict[str, int],
    opts: Options,
    avoid: tuple[Feature, ...] = (),
) -> tuple[tuple[int, int], str] | None:
    """Pick the annotated cassette that usable sites bracket most tightly.

    Replacing a vector's existing insert is the ordinary case, and the
    vector file already says where that insert is.  Choosing the CDS whose
    flanking single-cutters sit closest to it avoids the failure mode of
    simply excising as much of the backbone as possible.

    `avoid` holds features the caller has already committed to keeping.
    Being the largest annotated CDS beside the cloning site is what puts
    a feature at the top of this list, and it is also exactly what makes
    a feature a fusion partner, so without this the two decisions
    contradict each other -- see pMAL-c5X, whose 1101 bp MBP was being
    fused behind and excised in the same design.
    """
    positions = sorted(cuts.values())
    if not positions:
        return None
    # Where the vector marks a cloning site, the cassette worth
    # replacing is the one that site is for.  Without this the choice is
    # made on how tightly single-cutters happen to bracket a feature,
    # which picks backbone genes: excluding pMAL-c5X's MBP left `rop`,
    # 192 bp of copy-number control 2.3 kb from the polylinker, as the
    # thing to replace.  Across the validation set a real cassette sits
    # 0 to 55 bp from the cloning site and the nearest backbone gene is
    # 324 bp away, so the two do not overlap.
    marked = annotated_polylinker(vector)
    best: tuple[int, tuple[int, int], str] | None = None
    for feat in vector.features:
        if feat.kind != "CDS" or len(feat) < 150:
            continue
        if any(feat.start < a.end and a.start < feat.end for a in avoid):
            continue
        if marked is not None:
            gap = max(0, marked[0] - feat.end, feat.start - marked[1])
            if gap > CASSETTE_REACH:
                continue
        before = [p for p in positions if p <= feat.start]
        after = [p for p in positions if p >= feat.end]
        if not before or not after:
            continue
        loss = (feat.start - max(before)) + (min(after) - feat.end)
        if loss > 2 * opts.max_flank_loss:
            continue
        if best is None or loss < best[0]:
            best = (loss, (feat.start, feat.end), feat.name)
    if best is None:
        return None
    return best[1], best[2]


# What a vector calls its cloning site when it bothers to annotate one.
_MCS_NAMES = ("mcs", "multiple cloning site", "polylinker", "cloning site")


def annotated_polylinker(vector: Record | None) -> tuple[int, int] | None:
    """The cloning site the vector's own map marks out, if it marks one."""
    if vector is None:
        return None
    for f in vector.features:
        name = f.name.lower().strip()
        if any(tag in name for tag in _MCS_NAMES):
            return f.start, f.end
    return None


def _find_polylinker(
    unique: dict[str, int],
    clean: dict[str, int],
    span: int,
    vector: Record | None = None,
) -> tuple[int, int] | None:
    """Locate the vector's polylinker.

    An annotated cloning site wins outright when the vector has one.
    Inferring the site from clustered single-cutters is a fallback for
    maps that say nothing, and it over-reaches on a crowded vector:
    pET-28a annotates its MCS as 158-203, and inference stretched that to
    158-401, which swallowed the ribosome binding site, the start codon,
    the 6xHis tag and the thrombin site.  Cutting there deletes the whole
    N-terminal expression apparatus, and every one of those features is
    named on the map that was not being read.

    When no annotation exists the cluster is scored on every site that is
    unique in the vector, not only the ones this insert happens to leave
    usable.  A polylinker is a property of the vector: filtering by the
    insert first can strip a real nine-site cloning site down to three and
    let some incidental trio of sites elsewhere in the backbone outrank it.
    """
    marked = annotated_polylinker(vector)
    if marked is not None:
        lo, hi = marked
        inside = [p for p in clean.values() if lo <= p <= hi]
        if len(inside) >= 2:
            return min(inside), max(inside)

    pts = sorted(unique.values())
    if len(pts) < 2:
        return None
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    for i in range(len(pts)):
        j = i
        while j + 1 < len(pts) and pts[j + 1] - pts[i] <= span:
            j += 1
        lo, hi = pts[i], pts[j]
        usable = sum(1 for p in clean.values() if lo <= p <= hi)
        if usable < 2:
            continue
        key = (j - i + 1, usable)
        if best is None or key > best[0]:
            best = (key, (lo, hi))
    return best[1] if best else None


def _insertion_pairs(
    clean: dict[str, int],
    unique: dict[str, int],
    vector: Record,
    opts: Options,
    issues: list[Issue],
) -> tuple[list[EnzymePair], list[Issue]]:
    """Enzyme pairs for dropping an insert into an empty polylinker.

    A polylinker is a cluster of single-cutters packed into a few dozen
    bases, so the two chosen sites are close together and the piece removed
    between them is tiny.  Wider separation is preferred, because once one
    enzyme has cut, the other site sits that far from a fragment end and
    cuts poorly when it is too close.
    """
    found = _find_polylinker(unique, clean, opts.polylinker_span, vector)
    if found is None:
        in_cluster = clean
        issues.append(
            Issue(
                "warning",
                "no polylinker found: the usable single-cutters are spread "
                "across the vector, so the chosen sites may not sit in a "
                "cloning site. Check them against your map.",
            )
        )
    else:
        lo, hi = found
        in_cluster = {n: p for n, p in clean.items() if lo <= p <= hi}
        blocked = sum(1 for p in unique.values() if lo <= p <= hi) - len(in_cluster)
        issues.append(
            Issue(
                "note",
                f"inserting into the polylinker at {lo + 1}-{hi + 1}; "
                f"{len(in_cluster)} of its sites are usable"
                + (f", {blocked} also cut the insert" if blocked else "")
                + "; no vector cassette is being replaced",
            )
        )

    pairs: list[EnzymePair] = []
    for n5, p5 in in_cluster.items():
        e5 = ez.ENZYMES[n5]
        cut5 = p5 + e5.top
        for n3, p3 in in_cluster.items():
            e3 = ez.ENZYMES[n3]
            cut3 = p3 + e3.top
            excision = cut3 - cut5
            if excision <= 0 or excision > opts.max_insert_excision:
                continue
            if ez.compatible_ends(e5, e3):
                continue
            reasons: list[str] = []
            score = 0.0
            # Separation helps the double digest, but only up to a point;
            # past that it is just vector being thrown away.
            score += min(excision, 120) / 10.0
            score -= max(excision - 120, 0) / 40.0
            if excision < opts.close_site_warning:
                score -= 8
                reasons.append(f"sites only {excision} bp apart")
            if e5.is_blunt or e3.is_blunt:
                score -= 20
                reasons.append("one end is blunt, ligation is less efficient")
            if e5.is_degenerate or e3.is_degenerate:
                score -= 15
                reasons.append("degenerate recognition site")
            score -= (e5.rank + e3.rank) / 10.0
            pairs.append(
                EnzymePair(e5, e3, p5, p3, excision, score, reasons, "insert")
            )

    pairs.sort(key=lambda p: (-p.score, p.five.name, p.three.name))
    if not pairs:
        issues.append(
            Issue(
                "error",
                "no admissible enzyme pair in the polylinker (need two "
                "single-cutters, absent from the insert, with incompatible "
                "ends)",
            )
        )
    return pairs, issues


def candidate_pairs(
    insert: Record, vector: Record, opts: Options
) -> tuple[list[EnzymePair], list[Issue]]:
    issues: list[Issue] = []
    unique = ez.unique_cutters(vector.seq, circular=vector.circular)
    if not unique:
        issues.append(Issue("error", "no single-cutter sites found in the vector"))
        return [], issues

    clean = {
        name: pos
        for name, pos in unique.items()
        if not ez.find_sites(insert.seq, ez.ENZYMES[name].site)
    }
    if not clean:
        issues.append(
            Issue(
                "error",
                "every single-cutter in the vector also cuts the insert; "
                "consider a different vector region or a seamless method",
            )
        )
        return [], issues

    # Going through a shuttle adds a third sequence the enzymes must not
    # cut.  A pair that is clean in the insert and unique in the
    # destination can still carve up the shuttle backbone in step two.
    if opts.shuttle_record is not None:
        spared = {
            name: pos
            for name, pos in clean.items()
            if not ez.find_sites(
                opts.shuttle_record.seq,
                ez.ENZYMES[name].site,
                circular=opts.shuttle_record.circular,
            )
        }
        if len(spared) >= 2:
            dropped = sorted(set(clean) - set(spared))
            if dropped:
                issues.append(
                    Issue(
                        "note",
                        f"{len(dropped)} enzyme(s) excluded because they cut "
                        f"the shuttle {opts.shuttle_record.name}: "
                        + ", ".join(dropped[:8])
                        + ("..." if len(dropped) > 8 else ""),
                    )
                )
            clean = spared
        else:
            issues.append(
                Issue(
                    "warning",
                    "too few enzymes avoid the shuttle backbone, so the "
                    "choice was made without that constraint; check the "
                    "step-2 digest fragments carefully",
                )
            )

    # Work out what region of the vector is being replaced.
    target: tuple[int, int] | None = None
    if opts.replace:
        feat = vector.find_feature(opts.replace)
        if feat is None:
            issues.append(
                Issue("error", f"vector has no feature matching {opts.replace!r}")
            )
            return [], issues
        target = (feat.start, feat.end)
        issues.append(
            Issue("note", f"replacing annotated feature {feat.describe()}")
        )
    elif opts.window:
        target = opts.window
    elif opts.mode == "cfusion":
        # A C-terminal fusion reads into something, so that something
        # has to survive.  Left to itself the cassette finder picks the
        # fusion partner -- it is the largest annotated CDS near the
        # cloning site, which is exactly what makes it the partner -- and
        # the design then reports reading through into a feature it has
        # just deleted.  So no cassette is replaced in this mode: the
        # insert goes into the polylinker in front of the partner.
        partner = downstream_partner(vector)
        if partner is not None:
            issues.append(
                Issue(
                    "note",
                    f"no cassette is being replaced: {partner.feature.name} "
                    "is the fusion partner and has to stay. The insert goes "
                    "into the polylinker in front of it",
                )
            )
    else:
        keep = (
            _fusion_leader_features(vector)
            if opts.mode == "nfusion"
            else ()
        )
        auto = _auto_cassette(vector, clean, opts, avoid=keep)
        if auto is not None:
            target, name = auto
            issues.append(
                Issue(
                    "note",
                    f"replacing annotated cassette {name} "
                    f"({target[0] + 1}-{target[1]}); pass --replace to choose "
                    "a different one",
                )
            )
        elif keep:
            names = ", ".join(f.name for f in keep)
            issues.append(
                Issue(
                    "note",
                    f"no cassette is being replaced: {names} "
                    + ("is" if len(keep) == 1 else "are")
                    + " what the insert fuses behind and "
                    + ("has" if len(keep) == 1 else "have")
                    + " to stay. The insert goes into the polylinker after "
                    + ("it" if len(keep) == 1 else "them"),
                )
            )

    if target is not None and opts.mode in ("cfusion", "nfusion"):
        # The user named a region explicitly. If it is what the fusion
        # is built on, the two instructions contradict each other and
        # guessing which was meant is not this function's business.
        if opts.mode == "cfusion":
            partner = downstream_partner(vector)
            fused_to = (partner.feature,) if partner is not None else ()
            side, where = "C-terminal", "in front of"
        else:
            fused_to = _fusion_leader_features(vector)
            side, where = "N-terminal", "after"
        clash = [
            f for f in fused_to
            if target[0] < f.end and f.start < target[1]
        ]
        if clash:
            names = ", ".join(f.name for f in clash)
            issues.append(
                Issue(
                    "error",
                    f"an {side} fusion to {names} cannot also replace "
                    f"{'it' if len(clash) == 1 else 'them'}: the region "
                    f"{target[0] + 1}-{target[1]} overlaps "
                    + ", ".join(f.describe() for f in clash)
                    + f". Either clone {where} "
                    + ("it" if len(clash) == 1 else "them")
                    + " without --replace, or use --mode native to replace "
                    + ("it" if len(clash) == 1 else "them")
                    + " and give up the fusion",
                )
            )
            return [], issues

    if target is None:
        # Nothing to replace: this is an insertion into an empty polylinker,
        # where the sites sit tens of bases apart and almost no vector is
        # removed.  That is the ordinary case for a cloning vector and must
        # not be judged by the cassette-replacement rules.
        return _insertion_pairs(clean, unique, vector, opts, issues)

    tstart, tend = target
    pairs: list[EnzymePair] = []
    for n5, p5 in clean.items():
        e5 = ez.ENZYMES[n5]
        cut5 = p5 + e5.top
        if cut5 > tstart or tstart - cut5 > opts.max_flank_loss:
            continue
        for n3, p3 in clean.items():
            e3 = ez.ENZYMES[n3]
            cut3 = p3 + e3.top
            if cut3 < tend or cut3 - tend > opts.max_flank_loss:
                continue
            excision = cut3 - cut5
            if excision < max(opts.min_excision, 1):
                continue
            if ez.compatible_ends(e5, e3):
                continue  # ends would religate, losing directionality
            reasons: list[str] = []
            score = 0.0
            # Vector sequence lost either side of the cassette.
            score -= (tstart - cut5) / 20.0
            score -= (cut3 - tend) / 20.0
            if e5.is_blunt or e3.is_blunt:
                score -= 20
                reasons.append("one end is blunt, ligation is less efficient")
            if e5.is_degenerate or e3.is_degenerate:
                score -= 15
                reasons.append("degenerate recognition site")
            score -= (e5.rank + e3.rank) / 10.0
            pairs.append(
                EnzymePair(e5, e3, p5, p3, excision, score, reasons, "replace")
            )

    pairs.sort(key=lambda p: (-p.score, p.five.name, p.three.name))
    if not pairs:
        issues.append(
            Issue(
                "error",
                "no admissible enzyme pair brackets the target region "
                "(need two single-cutters, absent from the insert, with "
                "incompatible ends, within "
                f"{opts.max_flank_loss} bp of the cassette)",
            )
        )
    return pairs, issues


def _forced_pair(
    vector: Record, insert: Record, opts: Options
) -> tuple[EnzymePair | None, list[Issue]]:
    issues: list[Issue] = []
    e5 = ez.get(opts.five_enzyme)  # type: ignore[arg-type]
    e3 = ez.get(opts.three_enzyme)  # type: ignore[arg-type]
    for enz in (e5, e3):
        hits = ez.find_sites(vector.seq, enz.site, circular=vector.circular)
        if len(hits) == 0:
            issues.append(Issue("error", f"{enz.name} does not cut the vector"))
        elif len(hits) > 1:
            issues.append(
                Issue(
                    "error",
                    f"{enz.name} cuts the vector {len(hits)} times "
                    f"(at {', '.join(str(h + 1) for h in hits)}); "
                    "it cannot be used for directional cloning here",
                )
            )
        internal = ez.find_sites(insert.seq, enz.site)
        if internal:
            issues.append(
                Issue(
                    "error",
                    f"{enz.name} cuts the insert at "
                    f"{', '.join(str(i + 1) for i in internal)}",
                )
            )
    if any(i.level == "error" for i in issues):
        return None, issues
    p5 = ez.find_sites(vector.seq, e5.site, circular=vector.circular)[0]
    p3 = ez.find_sites(vector.seq, e3.site, circular=vector.circular)[0]
    if p5 > p3:
        issues.append(
            Issue(
                "warning",
                f"{e5.name} (vector position {p5 + 1}) lies downstream of "
                f"{e3.name} ({p3 + 1}); the excised region wraps the origin",
            )
        )
    if ez.compatible_ends(e5, e3):
        issues.append(
            Issue(
                "warning",
                f"{e5.name} and {e3.name} leave compatible ends, so the "
                "vector can religate on itself and the insert can go in "
                "backwards",
            )
        )
    excision = (p3 + e3.top) - (p5 + e5.top)
    return EnzymePair(e5, e3, p5, p3, excision, 0.0), issues


# ---------------------------------------------------------------------------
# annealing regions
# ---------------------------------------------------------------------------


# Alternative 5' pads.  Every Kozak variant here keeps a purine at -3,
# the position that actually matters for initiation; they differ further
# out, which is enough to break a stem without weakening translation.
KOZAK_VARIANTS: list[tuple[str, str]] = [
    ("GCCACC", "consensus Kozak"),
    ("GCCGCC", "Kozak, G at -3"),
    ("ACCACC", "Kozak, A at -6"),
    ("CCACC", "5 nt Kozak"),
    ("GCCACA", "Kozak, A at -1"),
    ("", "no Kozak"),
]

# Spare spacers, all AT-leaning and low-structure, for when the default
# contributes to a stem.
SPACER_VARIANTS: list[str] = ["TGACTG", "TTAGCA", "ACTATG", "CAATGA"]


def _score_anneal(seq: str, opts: Options) -> float:
    """Quality of the template-binding region, ignoring the tail."""
    tm = thermo.tm(seq, opts.conditions)
    gc = gc_fraction(seq)
    penalty = abs(tm - opts.target_tm)
    if not 0.40 <= gc <= 0.60:
        penalty += 6 * min(abs(gc - 0.40), abs(gc - 0.60)) * 10
    if thermo.longest_run(seq) >= 4:
        penalty += 6
    clamp = thermo.gc_clamp(seq)
    if clamp == 0:
        penalty += 5
    elif clamp >= 4:
        penalty += 4
    if seq[-1] not in "GC":
        penalty += 1.5
    if thermo.three_prime_dg(seq) < -6.0:
        penalty += 4
    return penalty


def score_oligo(o: Oligo, opts: Options) -> float:
    """Whole-oligo score: annealing quality plus secondary structure.

    Structure has to be judged on the full oligo, because the tail is part
    of what folds; the DR4 forward primer's stem straddles the restriction
    site and the Kozak, and is invisible if you only look at the
    template-binding region.
    """
    return _score_anneal(o.anneal, opts) + opts.structure_weight * o.structure_penalty


def _oligo_candidates(
    template: str,
    opts: Options,
    *,
    name: str,
    site: str,
    pad: str,
    spacer: str,
    reverse: bool,
) -> list[Oligo]:
    out: list[Oligo] = []
    for length in range(opts.anneal_min, opts.anneal_max + 1):
        if length > len(template):
            break
        sub = template[-length:] if reverse else template[:length]
        region = revcomp(sub) if reverse else sub
        if set(region) - set("ACGT"):
            continue
        out.append(
            Oligo(
                name=name,
                spacer=spacer,
                site=site,
                pad=pad,
                anneal=region,
                conditions=opts.conditions,
            )
        )
    return out


def _pick_matched(
    fwd: list[Oligo], rev: list[Oligo], opts: Options
) -> tuple[Oligo, Oligo]:
    """Choose the pair whose annealing temperatures match each other."""
    fwd = sorted(fwd, key=lambda o: score_oligo(o, opts))[:12]
    rev = sorted(rev, key=lambda o: score_oligo(o, opts))[:12]
    best: tuple[float, Oligo, Oligo] | None = None
    for f in fwd:
        fs = score_oligo(f, opts)
        for r in rev:
            combined = (
                fs
                + score_oligo(r, opts)
                + 4.0 * abs(f.tm_anneal - r.tm_anneal)
            )
            if best is None or combined < best[0]:
                best = (combined, f, r)
    if best is None:
        raise ValueError("no usable annealing region; insert may be too short")
    return best[1], best[2]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


# What a leader is called, used only to describe it.  The mode is
# decided by where a feature sits, never by its name, so a leader none
# of these words match still fuses correctly -- it is just described in
# general terms.  Signal peptides are checked first because they are
# the one kind that does not end up on the product.
_SIGNAL_WORDS = (
    "signal sequence", "signal peptide", "secretion signal",
    "pelb", "ompa", "ompt", "phoa", "dsba", "tora", "stii",
)
_PROTEASE_WORDS = (
    "thrombin", "factor xa", "tev", "prescission", "enterokinase",
    "hrv 3c", "3c site",
)
_TAG_WORDS = ("his", "tag", "flag", "myc", "strep", "gst", "mbp", "ha")


# How far a leader's last base may sit from the cloning site and still
# be a leader.  A translational leader runs up to the insert -- whatever
# lies between the two is translated as well, so anything that is really
# a leader ends adjacent to the polylinker, or a few bases into its
# annotation.  Across the validation set the gap is 0 to 3 bases, and
# three vectors overlap it slightly.
#
# The check is here because `fusion_frame` anchors on any CDS that
# begins at a candidate start codon, and two of those are not leaders at
# all: pVT50's AmpR starts at one 1200 bp from the cloning site, and
# pLJM1-EGFP's EGFP cassette encloses the cut sites rather than
# preceding them -- it is the thing being replaced. Both were read as
# native before, but only because their names did not look like affinity
# tags, which is not a reason and did not survive asking pET-22b.
LEADER_REACH = 60

# The same question on the 3' side: how far past the cloning site a
# coding sequence may begin and still be something the insert reads
# into.  pEGFP-N1's EGFP starts 7 bp after its MCS; the C-terminal
# 6xHis on the pET vectors starts at 0 to 4.
PARTNER_REACH = 60

# Below this a downstream CDS is a tag you opt into, not the reason the
# vector exists.  Switching mode on its own account would silently
# strip the insert's stop codon to add a few residues nobody asked for,
# so a short one is reported and left alone, while a partner the size
# of a fluorescent protein decides the mode.  Every affinity tag in the
# validation set is 18 bp; EGFP is 720.
PARTNER_MIN_CODONS = 50


def _leader_kind(f: Feature) -> str:
    """What a feature between the start codon and the cloning site is."""
    name = f.name.lower()
    if any(w in name for w in _SIGNAL_WORDS):
        return "signal"
    if any(w in name for w in _PROTEASE_WORDS):
        return "protease"
    if any(w in name for w in _TAG_WORDS):
        return "tag"
    return "other"


def choose_mode(vector: Record) -> tuple[str, str]:
    """Native or fusion, decided from what the vector is built for.

    A vector carrying its own start codon with coding sequence in frame
    between it and the cloning site wants a fusion: its ribosome binding
    site is positioned for *its* ATG, and bringing your own leaves the
    start codon too far from the RBS to be used.  pET-28a is the worked
    example -- its ATG sits 8 nt from the RBS, and an insert's own lands
    109 nt away.  A vector with no such apparatus wants native cloning.

    What makes a leader is where it sits, not what it is called: in
    frame, after the start codon, and running up to the cloning site.
    An earlier version asked instead whether any confirmed feature was
    named like an affinity tag, which read pET-22b -- whose leader is
    the pelB signal sequence, confirmed, in frame and 8 nt after the
    RBS -- as having "no tag to fuse to".  It fell back to native and
    then refused all six validation genes, because an insert's own ATG
    lands 78-100 nt from that vector's RBS.  Matching on names failed
    closed on the one bacterial vector in the set whose leader is not
    an affinity tag.
    """
    frame = fusion_frame(vector)
    marked = annotated_polylinker(vector)
    leaders: list[Feature] = []
    if frame is None or not frame.is_confirmed:
        why = "the vector offers no confirmed upstream reading frame"
    elif marked is None:
        why = (
            "the vector annotates no cloning site, so nothing establishes a "
            "leader running up to one"
        )
    else:
        leaders = [f for f in frame.confirmed_by if f.start < marked[0]]
        reach = max((f.end for f in leaders), default=None)
        if reach is None or marked[0] - reach > LEADER_REACH:
            gap = (
                ""
                if reach is None
                else f", the nearest ending {marked[0] - reach} bp short"
            )
            why = (
                "the vector has a start codon but no coding sequence running "
                f"up to its cloning site{gap}"
            )
            leaders = []
        else:
            why = ""
    if not leaders:
        # No leader to fuse behind, so the insert brings its own start
        # codon.  That leaves the other kind of fusion: a vector whose
        # cloning site sits in front of something, which the insert
        # reads into once its stop codon is removed.  pEGFP-N1 is the
        # worked example -- EGFP begins 7 bp past the MCS -- and native
        # cloning there is not an error, it simply declines the fusion
        # the vector exists for and says nothing about it.
        partner = downstream_partner(vector)
        if partner is not None and partner.decides_the_mode:
            return "cfusion", (
                f"the vector's cloning site runs into {partner.feature.name} "
                f"{partner.gap} bp downstream, in a vector with no leader of "
                "its own, so the insert supplies the start codon and reads "
                f"through into {partner.feature.name}. Its stop codon is "
                "removed to let it"
            )
        return "native", why
    frame = replace(frame, confirmed_by=tuple(leaders))
    positioned = (
        ", which is what its ribosome binding site is positioned for"
    )
    signals = [f.name for f in frame.confirmed_by if _leader_kind(f) == "signal"]
    if signals:
        # Worth saying outright: a secretion signal is a fusion in the
        # arithmetic but not in the product.  Treating it as an
        # N-terminal tag would describe a protein that is never made.
        return "nfusion", (
            "the vector carries " + ", ".join(signals) + " in frame "
            "downstream of its own start codon" + positioned
            + ". That leader is a secretion signal rather than an affinity "
            "tag: it is cleaved on export, so it does not stay on the "
            "mature protein"
        )
    named = [
        f.name for f in frame.confirmed_by if _leader_kind(f) in ("tag", "protease")
    ]
    return "nfusion", (
        "the vector carries "
        + ", ".join(named or [f.name for f in frame.confirmed_by])
        + " in frame downstream of its own start codon" + positioned
    )


@dataclass(frozen=True)
class VectorView:
    """A vector as this module reads it, in the frame its cassette uses.

    Every vector-reading function here was written for a cassette that
    runs forward, because `design` turns the vector round once at the top
    and everything after that point can assume it.  Called directly on a
    vector whose cassette reads against the map the same functions answer
    about the wrong strand, and they do it quietly: on pET-28a,
    `fusion_frame` comes back anchored on the start of KanR and
    `choose_mode` comes back native, which is the opposite of the truth.

    So this is the composed reading, and the only one a caller outside
    this module should use.  One question -- what is this vector? -- with
    one answer, instead of seven that can disagree with each other.
    """

    # The vector as it was read, in the coordinates the user's map uses.
    vector: Record
    # The strand every other field below is about.
    reading: Record
    flipped: bool
    direction: int
    host: str | None
    promoter: Feature | None
    frame: FusionFrame | None
    partner: DownstreamPartner | None
    polylinker: tuple[int, int] | None
    mode: str
    why: str
    # A signal peptide the vector itself supplies, pET-22b's pelB being
    # the one in the validation set.  Kept distinct from the tags: pelB
    # is cleaved on export, so it is not a tag on the product, and an
    # insert that carries its own signal sequence would be going in
    # behind a second one.
    signal_leader: Feature | None = None

    def on_the_map(self, position: int) -> int:
        """A position in this reading, in the coordinates of the map."""
        if not self.flipped:
            return position
        return len(self.vector) - position


def inspect(vector: Record) -> VectorView:
    """Read a vector once, on the strand its cassette actually uses.

    Deliberately free of `Issue`s.  The flip is a fact about the reading,
    not a finding about the design, and it is already on the face of the
    result as `flipped`; `design` raises it as a note when it matters,
    which is when positions are about to be quoted back.
    """
    direction = expression_direction(vector)
    flipped = direction < 0
    reading = vector.reverse_complement() if flipped else vector
    mode, why = choose_mode(reading)
    signal = next(
        (f for f in _fusion_leader_features(reading)
         if _leader_kind(f) == "signal"),
        None,
    )
    return VectorView(
        vector=vector,
        reading=reading,
        flipped=flipped,
        direction=direction,
        host=detect_host(reading),
        promoter=driving_promoter(reading),
        frame=fusion_frame(reading),
        partner=downstream_partner(reading),
        polylinker=annotated_polylinker(reading),
        mode=mode,
        why=why,
        signal_leader=signal,
    )


def design(insert: Record, vector: Record, opts: Options | None = None) -> Design:
    opts = opts or Options()
    issues: list[Issue] = []

    # Turn the vector round once, up front, if its expression cassette
    # reads against the map.  Everything after this point can then assume
    # the cassette runs forward, which is what the rest of the module was
    # written for.  The flip is recorded so positions can be quoted back
    # in the coordinates the user's own map uses.
    if opts.mode == "auto":
        from dataclasses import replace as _replace

        probe = (
            vector.reverse_complement()
            if expression_direction(vector) < 0
            else vector
        )
        chosen, why = choose_mode(probe)
        opts = _replace(opts, mode=chosen)
        issues.append(
            Issue("note", f"cloning as {chosen}: {why}")
        )

    flipped = expression_direction(vector) < 0
    if flipped:
        original = vector
        vector = vector.reverse_complement()
        issues.append(
            Issue(
                "note",
                f"{original.name}'s expression cassette runs on the reverse "
                "strand, so the vector is shown reverse-complemented here "
                "and the insert goes in the direction the promoter reads. "
                f"Position p below is {len(original)} - p on your map",
            )
        )

    seq = insert.seq
    issues.extend(_check_orf(seq, opts.mode))

    if opts.five_enzyme and opts.three_enzyme:
        pair, pair_issues = _forced_pair(vector, insert, opts)
        alternatives: list[EnzymePair] = []
    else:
        ranked, pair_issues = candidate_pairs(insert, vector, opts)
        pair = ranked[0] if ranked else None
        alternatives = ranked[1:6]
    issues.extend(pair_issues)
    if pair is None:
        raise DesignFailed(issues)

    core_start, core_end = 0, len(seq)
    if opts.mode == "cfusion":
        if seq[-3:].upper() in STOP_CODONS:
            core_end -= 3
            issues.append(
                Issue("note", "stop codon removed for C-terminal fusion")
            )
        else:
            issues.append(
                Issue(
                    "warning",
                    "C-terminal fusion requested but the insert does not end "
                    "in a stop codon; nothing was removed",
                )
            )

    pad5, pad3, pad_issues = _frame_pads(
        pair, opts, core_end - core_start, seq, vector
    )
    issues.extend(pad_issues)

    core = seq[core_start:core_end]
    pad5_options = [pad5]
    if opts.optimize_pad and opts.mode == "native" and pad5:
        pad5_options = [p for p, _ in KOZAK_VARIANTS if p]
    forward_pool: list[Oligo] = []
    for candidate_pad in pad5_options:
        forward_pool += _oligo_candidates(
            core, opts, name=f"{insert.name}_F", site=pair.five.site,
            pad=candidate_pad, spacer=opts.spacer, reverse=False,
        )
    forward, reverse_oligo = _pick_matched(
        forward_pool,
        _oligo_candidates(
            core, opts, name=f"{insert.name}_R", site=pair.three.site,
            pad=pad3, spacer=opts.spacer, reverse=True,
        ),
        opts,
    )

    result = Design(
        insert=insert,
        vector=vector,
        pair=pair,
        forward=forward,
        reverse=reverse_oligo,
        options=opts,
        core_start=core_start,
        core_end=core_end,
        issues=issues,
        alternatives=alternatives,
        vector_flipped=flipped,
    )
    result.issues.extend(validate(result))
    return result


class DesignFailed(Exception):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues if i.level == "error"))


@dataclass
class OpenReadingFrame:
    """Whether a sequence reads as an uninterrupted coding sequence.

    One implementation of this question, used by the designer and by
    the source-organism checks.  There were two, with different logic
    and different severities, which is how a sequence could be called
    unusable by one and merely untidy by the other.
    """

    length: int
    starts_with_atg: bool
    ends_with_stop: bool
    in_frame: bool
    internal_stops: list[int] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.starts_with_atg
            and self.ends_with_stop
            and self.in_frame
            and not self.internal_stops
        )

    def faults(self) -> list[str]:
        out = []
        if not self.starts_with_atg:
            out.append("it does not begin with ATG")
        if not self.in_frame:
            out.append(f"its length ({self.length} nt) is not a multiple of 3")
        if self.internal_stops:
            where = ", ".join(str(c) for c in self.internal_stops[:5])
            out.append(
                f"{len(self.internal_stops)} stop codon(s) interrupt it, at "
                f"codon {where}"
            )
        if not self.ends_with_stop:
            out.append("it does not end in a stop codon")
        return out


def read_orf(seq: str) -> OpenReadingFrame:
    seq = seq.upper()
    codons = [seq[i : i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    internal = [
        index + 1
        for index, codon in enumerate(codons[:-1])
        if codon in STOP_CODONS
    ]
    return OpenReadingFrame(
        length=len(seq),
        starts_with_atg=seq.startswith("ATG"),
        ends_with_stop=bool(codons) and codons[-1] in STOP_CODONS,
        in_frame=len(seq) % 3 == 0,
        internal_stops=internal,
    )


def _check_orf(seq: str, mode: str = "native") -> list[Issue]:
    """Grade the insert's reading frame for the mode it will be cloned in.

    What counts as a fault depends on the mode, and treating them alike
    gets two of the three wrong.  In native cloning the insert carries
    the start codon, so not having one is fatal.  In an N-terminal
    fusion the *vector* supplies it and the insert is a fragment joined
    downstream -- warning there that the insert "does not start with
    ATG" is backwards, since the designer is simultaneously noting that
    an ATG, if present, becomes a spare methionine.  A C-terminal fusion
    removes the stop codon deliberately, so missing one is the plan.

    A length that is not a multiple of three is fatal in every mode: it
    cannot encode a protein by itself, and in a fusion it frameshifts
    everything downstream of the insert, including the vector's own
    C-terminal tag.
    """
    out: list[Issue] = []
    orf = read_orf(seq)

    if not orf.starts_with_atg:
        if mode == "native":
            out.append(
                Issue(
                    "error",
                    "the insert does not start with ATG, and in native "
                    "cloning it is the only thing that could supply a start "
                    "codon. Clone as a fusion if the vector provides one, or "
                    "check the sequence is the coding strand and begins at "
                    "the start of the ORF",
                )
            )
        else:
            out.append(
                Issue(
                    "note",
                    "the insert does not start with ATG, which is expected "
                    "in a fusion: the vector supplies the start codon",
                )
            )

    if not orf.in_frame:
        out.append(
            Issue(
                "error",
                f"the insert is {orf.length} nt, which is not a multiple of "
                "3, so it cannot encode a protein in one frame"
                + (
                    " and would frameshift everything downstream of it, "
                    "including the vector's own C-terminal tag"
                    if mode != "native"
                    else ""
                ),
            )
        )
    elif not orf.ends_with_stop and mode != "cfusion":
        out.append(Issue("note", "insert does not end in a stop codon"))

    # Checked whatever the length.  Tying this to a clean length meant
    # the worst sequences escaped the strongest check: a genomic region
    # with fifteen in-frame stops whose length was not a multiple of
    # three got a mild warning about the length and nothing about the
    # stops, which are the decisive evidence.
    if orf.internal_stops:
        where = ", ".join(str(c) for c in orf.internal_stops[:5])
        more = (
            f" and {len(orf.internal_stops) - 5} more"
            if len(orf.internal_stops) > 5
            else ""
        )
        out.append(
            Issue(
                "error",
                f"{len(orf.internal_stops)} in-frame stop codon(s) interrupt "
                f"the insert, at codon {where}{more}. This is not a coding "
                "sequence: a genomic region still carrying introns and "
                "untranslated regions looks exactly like this, and so does "
                "a sequence read in the wrong frame or on the wrong strand",
            )
        )
    return out


@dataclass(frozen=True)
class FusionFrame:
    """The vector's reading frame, and the evidence it is the right one.

    `start` is the first base of the vector's own start codon.  Finding
    it is easy; being sure of it is the part that matters, because one
    base out frameshifts the entire protein and nothing downstream looks
    wrong.  So the frame is checked against the tags that sit between
    the start codon and the cloning site: each of them should begin a
    whole number of codons after the ATG.  Three features agreeing is
    evidence; none agreeing means the start codon is not where the
    annotation says.
    """

    start: int
    source: str
    confirmed_by: tuple[Feature, ...] = ()
    conflicting: tuple[Feature, ...] = ()
    # A coding feature that begins exactly at the start codon.  Not all
    # vectors annotate the ATG separately: pET-21b does not, and its
    # start codon is simply the first codon of the T7 tag, whose own
    # translation begins with that methionine.  An annotated CDS
    # starting at an ATG that independently sits the right distance from
    # the ribosome binding site is corroboration, not circularity --
    # two different pieces of the map agreeing.
    anchored_on: str = ""

    @property
    def is_confirmed(self) -> bool:
        if self.conflicting:
            return False
        return bool(self.confirmed_by) or bool(self.anchored_on)

    def describe(self) -> str:
        names = ", ".join(
            f"{f.name} at +{f.start - self.start}" for f in self.confirmed_by
        )
        return f"start codon at {self.start + 1} ({self.source}): {names}"


def _start_candidates(vector: Record) -> list[tuple[int, str]]:
    """Positions that could be the start codon, best-evidenced first.

    Three routes, because vectors mark this three different ways and
    none of them is universal.  pET-28a annotates the ATG as a feature
    of its own.  pET-21b does not, because its start codon is the first
    codon of the T7 tag.  pCold I annotates neither an ATG nor a
    ribosome binding site -- its start codon is the first codon of a
    translation-enhancing element, and the only thing that confirms it
    is that the tag and protease site downstream are in frame with it.

    Each route only proposes; `fusion_frame` is what decides, by
    checking the frame each candidate implies.
    """
    seq = vector.seq.upper()
    out: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(pos: int, source: str) -> None:
        if pos in seen or pos < 0 or seq[pos : pos + 3] != "ATG":
            return
        seen.add(pos)
        out.append((pos, source))

    sites = [
        f for f in vector.features
        if f.kind.lower() == "rbs" or "rbs" in f.name.lower()
    ]
    rbs = sites[0] if sites else None
    lo, hi = RBS_SPACING_USABLE

    def with_spacing(pos: int, source: str) -> str:
        if rbs is None:
            return source
        gap = pos - rbs.end
        return f"{source}, {gap} nt after the RBS" if lo <= gap <= hi else source

    # 1. An explicitly annotated start codon.
    for f in vector.features:
        if f.name.strip().upper() == "ATG" and len(f) == 3:
            add(f.start, "annotated")
        elif f.kind.lower() == "cds" and len(f) == 3:
            add(f.start, "annotated")

    # 2. A coding feature that begins with one.
    for f in sorted(vector.features, key=lambda f: f.start):
        if f.kind.lower() == "cds" and len(f) >= 6:
            add(f.start, with_spacing(f.start, f"the start of {f.name}"))

    # 3. The first ATG a usable distance after the ribosome binding
    #    site, when the map annotates one.
    if rbs is not None:
        for gap in range(lo, hi + 1):
            add(rbs.end + gap, f"first ATG {gap} nt after the RBS")
    return out


def start_codon_candidates(vector: Record, limit: int = 6) -> list[int]:
    """Every position that could plausibly be the vector's start codon.

    Printed when the frame cannot be settled, so the answer to a refusal
    is picking one of these rather than working out a coordinate by
    hand.  That matters most on a vector that had to be turned round:
    the position on the user's own map is not the one `--vector-frame-5`
    wants, and the first base of a codon read one way is the last base
    read the other.
    """
    seq = vector.seq.upper()
    found: list[int] = []
    for f in vector.features:
        if f.name.strip().upper() == "ATG" and len(f) == 3:
            found.append(f.start)
    sites = [
        f for f in vector.features
        if f.kind.lower() == "rbs" or "rbs" in f.name.lower()
    ]
    if sites:
        lo, hi = RBS_SPACING_USABLE
        for gap in range(lo, hi + 1):
            at = sites[0].end + gap
            if seq[at : at + 3] == "ATG" and at not in found:
                found.append(at)
    return found[:limit]



def _frame_hint(vector: Record) -> str:
    """How to answer a refusal, in the coordinates the flag expects."""
    candidates = start_codon_candidates(vector)
    if not candidates:
        return (
            " No candidate start codon was found either, so the frame has "
            "to come from the vector's documentation"
        )
    return (
        " Candidate start codons in the frame this design is working in: "
        + ", ".join(str(c + 1) for c in candidates)
        + ". Pass one of those to --vector-frame-5 -- they are already in "
        "the right frame, so no conversion from your own map is needed"
    )


def fusion_frame(vector: Record) -> FusionFrame | None:
    """Work out the vector's reading frame and check it against its tags.

    Only features between the start codon and the cloning site are used
    as evidence.  A tag downstream of the cloning site is deliberately
    ignored: its frame depends on the length of the insert that has not
    been put in yet, so it says nothing about the vector's own frame and
    would read as a conflict every time.
    """
    marked = annotated_polylinker(vector)

    def evaluate(start: int, source: str) -> FusionFrame:
        limit = marked[0] if marked else start + 400
        confirmed: list[Feature] = []
        conflicting: list[Feature] = []
        anchored_on = ""
        for f in vector.features:
            if not start <= f.start < limit:
                continue
            if f.kind.lower() != "cds" or len(f) < 6:
                continue
            if f.start == start:
                # The coding feature the start codon belongs to, rather
                # than one downstream of it.
                anchored_on = f.name
                confirmed.append(f)
            elif (f.start - start) % 3 == 0:
                confirmed.append(f)
            else:
                conflicting.append(f)
        confirmed.sort(key=lambda f: f.start)
        conflicting.sort(key=lambda f: f.start)
        return FusionFrame(
            start, source, tuple(confirmed), tuple(conflicting), anchored_on
        )

    # Take the best-evidenced candidate that its own downstream features
    # agree with.  Proposing several and checking each is what lets the
    # three ways a vector can mark its start codon be handled without
    # ranking them by guesswork: the frame either holds up or it does
    # not, and that is decided per candidate rather than up front.
    best: FusionFrame | None = None
    for start, source in _start_candidates(vector):
        frame = evaluate(start, source)
        if frame.is_confirmed:
            return frame
        if best is None:
            best = frame
    return best


@dataclass(frozen=True)
class DownstreamPartner:
    """Coding sequence past the cloning site that the insert reads into.

    The mirror of `FusionFrame`, and a weaker claim.  Upstream, the
    start codon is a guess that the tags between it and the cloning
    site either confirm or contradict.  Downstream there is nothing to
    guess: the partner's own first base is a codon boundary by
    definition, and the only question is whether the insert reaches it
    in frame, which is arithmetic the padding does.

    What is worth checking is that the partner is a whole number of
    codons.  An annotation that is not says the boundaries were drawn
    loosely, and the frame taken from its start may be a base or two
    out.
    """

    feature: Feature
    gap: int
    in_whole_codons: bool

    @property
    def start(self) -> int:
        return self.feature.start

    @property
    def is_a_tag(self) -> bool:
        return _leader_kind(self.feature) in ("tag", "protease")

    @property
    def decides_the_mode(self) -> bool:
        """Big enough, and not a tag, so the vector is built for it."""
        return (
            not self.is_a_tag
            and len(self.feature) >= 3 * PARTNER_MIN_CODONS
        )


def downstream_partner(vector: Record) -> DownstreamPartner | None:
    """The nearest coding sequence the insert could be fused to.

    Only features at or after the start of the cloning site count: a
    leader upstream of it is the other kind of fusion, and is
    `fusion_frame`'s business.  Nearest wins, because anything further
    off would have to be read through whatever lies between.
    """
    marked = annotated_polylinker(vector)
    if marked is None:
        return None
    best: DownstreamPartner | None = None
    for f in vector.features:
        if f.kind.lower() != "cds" or len(f) < 6:
            continue
        if f.start < marked[0] or f.start - marked[1] > PARTNER_REACH:
            continue
        found = DownstreamPartner(f, f.start - marked[1], len(f) % 3 == 0)
        if best is None or found.gap < best.gap:
            best = found
    return best


def _initiation_pad(
    opts: Options,
    seq: str,
    vector: Record | None,
    issues: list[Issue],
) -> str:
    """Translation initiation context, when the insert brings the ATG.

    Both native and C-terminal fusion need this and for the same
    reason: in neither does the vector supply a start codon, so the
    ribosome initiates on the insert's own and the bases in front of it
    are what decide how well.  Only an N-terminal fusion is exempt,
    because there the vector's ATG is the start and the insert's is
    just another methionine partway through.
    """
    if not (opts.add_kozak and seq.startswith("ATG")):
        return ""
    choice = opts.kozak
    if choice == "auto":
        host = detect_host(vector) if vector is not None else None
        if host is None:
            issues.append(
                Issue(
                    "warning",
                    "cannot tell from the vector's annotations which "
                    "host this expresses in, so no translation "
                    "initiation context was added. Pass --kozak "
                    "mammalian, yeast or bacterial if you know",
                )
            )
            choice = "none"
        elif host == "bacterial":
            # Not a missing Kozak: the wrong question.  Saying so
            # matters, because "no context added" looks identical
            # to the failure case above and is reached for an
            # entirely different reason.
            choice = "bacterial"
            issues.append(
                Issue(
                    "note",
                    "vector annotations look bacterial, so no Kozak "
                    "was added: bacteria do not use one. Translation "
                    "starts from the vector's own ribosome binding "
                    "site, and what matters is its distance from the "
                    "start codon rather than the bases just before it",
                )
            )
        else:
            choice = host
            issues.append(
                Issue(
                    "note",
                    f"vector annotations look {host}; added the "
                    f"{KOZAK_CONTEXTS[host][1]}",
                )
            )
    pad5, label = KOZAK_CONTEXTS.get(choice, KOZAK_CONTEXTS["none"])
    if pad5 and len(seq) > 3 and seq[3] not in "GA":
        issues.append(
            Issue(
                "note",
                f"{label} added, but the +4 base is {seq[3]} rather "
                "than G or A, so the context is strong but not optimal",
            )
        )
    return pad5


def _frame_pads(
    pair: EnzymePair,
    opts: Options,
    core_len: int,
    seq: str,
    vector: Record | None = None,
) -> tuple[str, str, list[Issue]]:
    """Kozak and frame-preserving padding for the chosen mode."""
    issues: list[Issue] = []
    pad5 = pad3 = ""

    if opts.mode == "native":
        return _initiation_pad(opts, seq, vector, issues), pad3, issues

    if opts.mode == "nfusion":
        frame5 = opts.vector_frame_5
        if frame5 is None and vector is not None:
            derived = fusion_frame(vector)
            if derived is None:
                issues.append(
                    Issue(
                        "error",
                        "N-terminal fusion needs a reading frame, and this "
                        "vector annotates neither a start codon nor a "
                        "ribosome binding site to find one from. Pass "
                        "--vector-frame-5 with the coordinate of a codon "
                        "start in the upstream tag."
                        + _frame_hint(vector),
                    )
                )
                return "", "", issues
            if derived.conflicting:
                # Refusing here is the point.  A frame that fails its own
                # check is worse than no frame, because the design it
                # produces looks entirely reasonable and encodes a
                # frameshifted protein.
                issues.append(
                    Issue(
                        "error",
                        f"the start codon at {derived.start + 1} does not "
                        "agree with the tags downstream of it: "
                        + ", ".join(
                            f"{f.name} begins {f.start - derived.start} nt "
                            f"after it, which is {(f.start - derived.start) % 3} "
                            "out of frame"
                            for f in derived.conflicting
                        )
                        + ". Either the annotation is wrong or this is not "
                        "the start codon the tags belong to."
                        + _frame_hint(vector),
                    )
                )
                return "", "", issues
            if not derived.confirmed_by:
                issues.append(
                    Issue(
                        "warning",
                        f"using the start codon at {derived.start + 1} "
                        f"({derived.source}), but there is no tag between it "
                        "and the cloning site to check the frame against, so "
                        "nothing confirms it",
                    )
                )
            else:
                downstream = [
                    f for f in derived.confirmed_by if f.start > derived.start
                ]
                if downstream:
                    evidence = (
                        f"{len(downstream)} feature(s) begin a whole number "
                        "of codons after it: "
                        + ", ".join(
                            f"{f.name} at +{f.start - derived.start}"
                            for f in downstream
                        )
                    )
                else:
                    evidence = (
                        f"it is the first codon of {derived.anchored_on}, "
                        "whose annotation and the ribosome binding site "
                        "agree on where translation starts"
                    )
                issues.append(
                    Issue(
                        "note",
                        f"reading frame taken from the start codon at "
                        f"{derived.start + 1} ({derived.source}); "
                        + evidence,
                    )
                )
            frame5 = derived.start
        if frame5 is None:
            issues.append(
                Issue(
                    "error",
                    "N-terminal fusion needs --vector-frame-5: the vector "
                    "coordinate of the first base of a codon in the upstream tag",
                )
            )
            return "", "", issues
        cut5 = pair.pos_five + pair.five.top
        carried = (cut5 - frame5) + (pair.five.length - pair.five.top)
        pad_len = (-carried) % 3
        pad5 = "GGA"[:pad_len] if pad_len else ""
        issues.append(
            Issue(
                "note",
                f"{carried} bases run from the start codon to the end of the "
                f"regenerated {pair.five.name} site, so "
                + (
                    "no padding is needed: the site is already in frame"
                    if pad_len == 0
                    else f"{pad_len} base(s) are added to bring the insert "
                    "into frame"
                ),
            )
        )
        return pad5, pad3, issues

    if opts.mode == "cfusion":
        # The insert carries the start codon here, exactly as in native
        # cloning -- what the vector supplies is the far end.
        pad5 = _initiation_pad(opts, seq, vector, issues)
        frame3 = opts.vector_frame_3
        if frame3 is None and vector is not None:
            partner = downstream_partner(vector)
            if partner is None:
                issues.append(
                    Issue(
                        "error",
                        "C-terminal fusion needs a reading frame, and this "
                        "vector annotates no coding sequence after its "
                        "cloning site to fuse to. Pass --vector-frame-3 "
                        "with the coordinate of a codon start in the "
                        "downstream partner.",
                    )
                )
                return "", "", issues
            if not partner.in_whole_codons:
                issues.append(
                    Issue(
                        "warning",
                        f"{partner.feature.name} is {len(partner.feature)} bp, "
                        "which is not a whole number of codons, so its "
                        "annotation does not mark a reading frame exactly "
                        "and the fusion may be a base or two out. Confirm "
                        "it against the vector's documentation, or pass "
                        "--vector-frame-3",
                    )
                )
            frame3 = partner.start
            issues.append(
                Issue(
                    "note",
                    "reading frame taken from "
                    f"{partner.feature.name} at {partner.start + 1}, which "
                    f"begins {partner.gap} bp past the cloning site",
                )
            )
        if frame3 is None:
            issues.append(
                Issue(
                    "error",
                    "C-terminal fusion needs --vector-frame-3: the vector "
                    "coordinate of the first base of a codon in the downstream tag",
                )
            )
            return "", "", issues
        cut3 = pair.pos_three + pair.three.top
        trailing = core_len + pair.three.top + (frame3 - cut3)
        pad_len = (-trailing) % 3
        pad3 = "GGA"[:pad_len] if pad_len else ""
        issues.append(
            Issue(
                "note",
                f"{trailing} bases run from the start of the insert to the "
                f"first codon of the fusion partner, so "
                + (
                    "no padding is needed: the partner is already in frame"
                    if pad_len == 0
                    else f"{pad_len} base(s) are added to bring it into frame"
                ),
            )
        )
        return pad5, pad3, issues

    issues.append(Issue("error", f"unknown mode {opts.mode!r}"))
    return "", "", issues


# ---------------------------------------------------------------------------
# trading melting temperature against secondary structure
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """One alternative primer, with what it costs and what it buys."""

    oligo: Oligo
    pad_label: str
    chosen: bool = False
    penalty: float = 0.0

    @property
    def tm(self) -> float:
        return self.oligo.tm_anneal

    @property
    def hairpin_dg(self) -> float:
        return min(self.oligo.hairpin.dg, 0.0)

    @property
    def dimer_dg(self) -> float:
        return min(self.oligo.self_dimer.dg, 0.0)

    @property
    def structure_dg(self) -> float:
        return self.oligo.structure_dg

    @property
    def structure_penalty(self) -> float:
        return self.oligo.structure_penalty

    @property
    def notes(self) -> str:
        bits = [self.pad_label]
        if self.oligo.spacer != SPACER_VARIANTS[0]:
            bits.append(f"spacer {self.oligo.spacer}")
        return ", ".join(b for b in bits if b)


def primer_alternatives(
    d: Design,
    which: str = "forward",
    *,
    limit: int = 10,
    vary_pad: bool = True,
    vary_spacer: bool = True,
) -> list[Variant]:
    """Rank alternative primers for one end, trading Tm against structure.

    The annealing region's start is fixed by where the coding sequence
    begins, so the levers are its length, the pad in front of it, and the
    spacer.  Each is varied independently and the results are ranked by
    how little they fold, with Tm reported alongside so the trade is
    visible rather than implied.
    """
    opts = d.options
    reverse = which == "reverse"
    current = d.reverse if reverse else d.forward
    site = (d.pair.three if reverse else d.pair.five).site
    core = d.core

    pads: list[tuple[str, str]]
    if vary_pad and not reverse and opts.mode == "native" and current.pad:
        pads = list(KOZAK_VARIANTS)
    elif vary_pad and not reverse and opts.mode == "native":
        pads = [("", "no Kozak"), *KOZAK_VARIANTS[:3]]
    else:
        pads = [(current.pad, "as designed")]

    spacers = SPACER_VARIANTS if vary_spacer else [current.spacer]

    seen: set[str] = set()
    variants: list[Variant] = []
    for pad, pad_label in pads:
        for spacer in spacers:
            for cand in _oligo_candidates(
                core, opts, name=current.name, site=site,
                pad=pad, spacer=spacer, reverse=reverse,
            ):
                if cand.sequence in seen:
                    continue
                seen.add(cand.sequence)
                variants.append(
                    Variant(
                        cand,
                        pad_label,
                        cand.sequence == current.sequence,
                        score_oligo(cand, opts),
                    )
                )

    # A spacer that changes nothing measurable is clutter, not an option.
    # Keep the default spacer's version of any outcome that is otherwise
    # identical, so the table shows real choices.
    deduped: dict[tuple, Variant] = {}
    for v in sorted(variants, key=lambda v: v.oligo.spacer != opts.spacer):
        key = (
            v.pad_label,
            len(v.oligo.anneal),
            round(v.tm, 2),
            round(v.structure_dg, 2),
        )
        if key not in deduped or v.chosen:
            deduped[key] = v
    variants = list(deduped.values())

    # Rank by the same score the designer uses, so --structure-weight
    # visibly moves this list.  Ties break towards the target Tm.
    variants.sort(key=lambda v: (v.penalty, abs(v.tm - opts.target_tm)))

    # Spread the table across the different kinds of choice rather than
    # filling it with one pad at fifteen lengths: a list that is all the
    # same decision shows no trade-off at all.
    per_pad = max(2, limit // max(len(pads), 1))
    counts: dict[str, int] = {}
    top: list[Variant] = []
    for v in variants:
        n = counts.get(v.pad_label, 0)
        if n >= per_pad:
            continue
        counts[v.pad_label] = n + 1
        top.append(v)
        if len(top) >= limit:
            break
    chosen = next((v for v in variants if v.chosen), None)
    if chosen is not None and chosen not in top:
        top.append(chosen)
    return top


# ---------------------------------------------------------------------------
# post-hoc validation
# ---------------------------------------------------------------------------


def collateral_losses(d: Design) -> list[Feature]:
    """Annotated features the excision destroys besides the one named.

    Replacing a cassette removes everything between the two cut sites,
    not only the feature the strategy is named after.  In an expression
    vector that span routinely holds a purification tag or a protease
    site sitting just past the cassette, and losing one is invisible
    until the protein will not bind the column.  Partial overlaps count:
    half a His tag is not a His tag.
    """
    if not d.vector.features:
        return []
    cut5 = d.pair.pos_five + d.pair.five.top
    cut3 = d.pair.pos_three + d.pair.three.top
    n = len(d.vector)

    span = (cut3 - cut5) % n

    def disrupted(f: Feature) -> bool:
        """Does the excision touch this feature at all?

        Both directions of overlap have to be tested.  Checking only
        whether the feature's ends fall inside the excision misses a
        feature that *contains* it, and that is not a corner case: a
        cloning site sits inside a screening marker by design, so
        pUC19's MCS is inside lacZ-alpha and cutting there disrupts a
        gene whose endpoints are nowhere near the cuts.
        """
        length = f.end - f.start
        starts_inside = (f.start - cut5) % n < span
        contains_cut = (cut5 - f.start) % n < length
        return starts_inside or contains_cut

    lost: list[Feature] = []
    for f in d.vector.features:
        if f.end - f.start >= n:
            continue
        # A cloning site is there to be cut: losing part of the MCS is
        # the operation, not a casualty of it.
        if any(tag in f.name.lower() for tag in _MCS_NAMES):
            continue
        if disrupted(f):
            lost.append(f)
    if d.pair.strategy == "replace":
        # The cassette being replaced on purpose is not a loss, and
        # neither is anything wholly inside it: that is the job.
        named = max(lost, key=len, default=None)
        return [f for f in lost if f is not named]
    # An insertion is supposed to remove almost nothing, so anything it
    # does remove is unintended by definition.  Skipping this case was a
    # real hole: cloning into pET-28a cut 132 bp past the annotated MCS
    # and deleted the ribosome binding site, the start codon, the 6xHis
    # tag and the thrombin site while reporting no losses at all.
    return lost


def _partner_on_the_product(d: Design) -> list[Issue]:
    """What a 3' fusion partner does to the product, outside nfusion.

    This exists because pEGFP-N1 reported a clean design with no issues
    at all.  Its EGFP begins 7 bp past the cloning site, and in native
    cloning the insert's own stop codon terminates translation before
    it, so the construct expresses the protein and no fluorescence --
    which is a working construct, and not the one the vector is for.
    Saying nothing let a design that quietly declines the whole point
    of the vector look identical to one that did not.

    Declining it is a legitimate choice, so native is a warning rather
    than an error: the construct works, it just is not a fusion.
    """
    partner = downstream_partner(d.vector)
    if partner is None:
        return []
    name = partner.feature.name
    if d.options.mode == "cfusion":
        return [
            Issue(
                "note",
                f"the product is a fusion: the insert's stop codon is gone "
                f"and translation runs on into the vector's {name}, "
                f"{len(partner.feature) // 3 - 1} residues on the C "
                "terminus. The insert supplies the start codon, so its own "
                "N terminus is unchanged",
            )
        ]
    if d.core[-3:].upper() not in STOP_CODONS:
        # No stop and no fusion frame derived for it: translation runs
        # into the vector on whatever frame the lengths happen to give.
        return [
            Issue(
                "warning",
                f"the insert has no stop codon and the vector's {name} "
                f"begins {partner.gap} bp past the cloning site, so "
                "translation runs on into it in whatever frame the lengths "
                "happen to leave. Use --mode cfusion to have the frame "
                "worked out, or keep a stop codon",
            )
        ]
    if partner.decides_the_mode:
        return [
            Issue(
                "warning",
                f"the vector's {name} begins {partner.gap} bp past the "
                "cloning site, and the insert's stop codon terminates "
                f"translation before it, so {name} is not on the product. "
                "That is what this vector is built to make. Use --mode "
                "cfusion for the fusion, or keep this if an untagged "
                "protein is what you want",
            )
        ]
    # A short C-terminal tag behind a stop codon is the ordinary case on
    # every pET vector and says nothing a native design needs to act on.
    # Only a partner the vector is built around, or a missing stop that
    # puts the frame in play, is worth raising here.
    return []


def tags_on_the_product(d: Design) -> list[Issue]:
    """Which of the vector's tags end up on the protein, and which do not.

    A vector like pET-28a offers a tag at each end, and which ones you
    actually get is not a property of the vector: it depends on your
    insert.  The N-terminal tag is translated because it sits between
    the start codon and the cloning site.  The C-terminal tag is only
    reached if the insert has no stop codon *and* the insert's length
    leaves the tag in frame -- so it comes down to the length of your
    gene modulo three, which is not something anyone should be working
    out by hand at the bench.
    """
    if d.options.mode != "nfusion":
        return _partner_on_the_product(d)
    frame = fusion_frame(d.vector)
    if frame is None:
        return []

    out: list[Issue] = []
    n = len(d.vector)
    cut3 = d.pair.pos_three + d.pair.three.top
    construct_len = len(d.construct)

    if frame.confirmed_by:
        # A secretion signal is fused to like a tag and then removed
        # like no tag is.  Reporting it among the N-terminal tags would
        # put a 22-residue leader on a product that never carries one;
        # leaving it out entirely would hide the export the vector
        # exists for.  So it gets its own note.
        signals = [
            f.name for f in frame.confirmed_by if _leader_kind(f) == "signal"
        ]
        if signals:
            out.append(
                Issue(
                    "note",
                    "the insert is fused to the vector's "
                    + ", ".join(signals)
                    + ", which directs the product to the periplasm and is "
                    "cut off by signal peptidase on the way, so it does not "
                    "appear on the mature protein. Residues the cloning site "
                    "contributes between the cleavage site and the insert do "
                    "remain",
                )
            )
        upstream = [
            f.name for f in frame.confirmed_by
            if _leader_kind(f) == "tag"
        ]
        if upstream:
            out.append(
                Issue(
                    "note",
                    "the product carries the vector's N-terminal "
                    + ", ".join(upstream)
                    + (", which sits" if len(upstream) == 1 else ", which sit")
                    + " between the start codon and the insert",
                )
            )

    downstream = [
        f
        for f in d.vector.features
        if f.kind.lower() == "cds"
        and 0 < (f.start - cut3) % n < 120
        and any(w in f.name.lower() for w in _TAG_WORDS)
    ]
    if not downstream:
        return out

    names = ", ".join(f.name for f in downstream)
    if d.core[-3:].upper() in STOP_CODONS:
        out.append(
            Issue(
                "note",
                f"the vector's C-terminal {names} is downstream of the "
                "insert's own stop codon, so it is not translated. To use "
                "it, remove the stop with --mode cfusion; to keep the "
                "protein untagged at its C terminus, leave it as it is",
            )
        )
        return out

    atg_at = (frame.start - cut3) % n
    for tag in downstream:
        gap = ((tag.start - cut3) % n - atg_at) % construct_len
        if gap % 3 == 0:
            out.append(
                Issue(
                    "note",
                    f"the insert has no stop codon and the vector's "
                    f"{tag.name} lands {gap} bases after the start codon, a "
                    "whole number of codons, so it is translated onto the C "
                    "terminus",
                )
            )
        else:
            out.append(
                Issue(
                    "error",
                    f"the insert has no stop codon, so translation runs on "
                    f"into the vector, but {tag.name} lands {gap} bases after "
                    f"the start codon, which is {gap % 3} out of frame. The C "
                    "terminus will be nonsense until a stop is reached. Keep "
                    "the insert's stop codon, or adjust the length by "
                    f"{3 - gap % 3} base(s)",
                )
            )
    return out


def screening_marker(d: Design) -> Feature | None:
    """The marker an insert disrupts to give a colour screen, if any.

    A cloning site sits inside lacZ-alpha on purpose, so that an insert
    interrupts it and the colony turns white. That is the readout, not
    damage, and the protocol needs to know so it can put X-gal and IPTG
    on the plate.
    """
    for f in collateral_losses(d):
        if "lacz" in f.name.lower():
            return f
    return None


def validate(d: Design) -> list[Issue]:
    out: list[Issue] = []
    opts = d.options

    out.extend(tags_on_the_product(d))

    lost = collateral_losses(d)
    # Disrupting the screening marker is not damage: it is the readout.
    # A cloning site sits inside lacZ-alpha precisely so that an insert
    # interrupts it, which is what turns a colony white.  Reporting that
    # as collateral loss would be technically true and practically
    # backwards, so it is separated out and the plate additives named.
    screening = [f for f in lost if "lacz" in f.name.lower()]
    lost = [f for f in lost if f not in screening]
    for marker in screening:
        out.append(
            Issue(
                "note",
                f"the insert interrupts {marker.name}, which is how "
                "blue/white screening works rather than a problem: plate on "
                "X-gal and IPTG and a colony carrying the insert is white "
                "while an empty religated vector is blue. It needs a host "
                "supplying the other half of the enzyme, which DH5-alpha "
                "and TOP10 both do",
            )
        )
    if lost:
        out.append(
            Issue(
                "warning",
                "the excision also removes "
                + ", ".join(f.describe() for f in lost)
                + ". Only the largest feature in the span is the cassette "
                "being replaced; the rest go with it. Check none of them is "
                "something the construct needs, such as a purification tag "
                "or a protease site",
            )
        )

    # Bacterial initiation: the vector's ribosome binding site has to end
    # up the right distance from the start codon.  This is the check that
    # replaces the Kozak check for E. coli, and it is about the vector's
    # geometry rather than anything in the primer.
    # Which start codon the ribosome actually uses depends on the mode,
    # and measuring the wrong one turns a correct fusion into a failure.
    # In native mode the insert brings its own ATG and that is what has
    # to sit near the RBS; in a fusion the vector's ATG is the start and
    # the insert's is just another Met partway through the protein.
    atg_at: int | None = None
    if detect_host(d.vector) == "bacterial":
        if d.options.mode == "native" and d.core.startswith("ATG"):
            cut5 = d.pair.pos_five + d.pair.five.top
            # Where the ATG lands, in vector coordinates: the cut, plus
            # what the forward primer's tail contributes downstream of it.
            carried = (d.pair.five.length - d.pair.five.top) + len(d.forward.pad)
            atg_at = cut5 + carried
        elif d.options.mode == "nfusion":
            frame = fusion_frame(d.vector)
            if frame is not None:
                atg_at = frame.start
                if d.core.startswith("ATG"):
                    # "after the tag" was wrong on any vector whose
                    # leader is not one -- on pET-22b the extra Met
                    # follows a signal peptide that is cleaved off.
                    leader = "the vector's leader"
                    if frame.confirmed_by:
                        leader = ", ".join(
                            f.name for f in frame.confirmed_by
                            if _leader_kind(f) != "protease"
                        ) or leader
                    out.append(
                        Issue(
                            "note",
                            "the insert keeps its own ATG, which becomes an "
                            f"extra methionine after {leader} rather than "
                            "the start codon. Harmless, but it is in the "
                            "product",
                        )
                    )
    if atg_at is not None:
        found = rbs_spacing(d.vector, atg_at)
        if found is None:
            out.append(
                Issue(
                    "warning",
                    "this looks like a bacterial expression vector but "
                    "annotates no ribosome binding site, so the distance "
                    "from the RBS to the start codon could not be checked. "
                    "That distance is what sets initiation in E. coli",
                )
            )
        else:
            gap, site = found
            lo, hi = RBS_SPACING_OPTIMAL
            usable_lo, usable_hi = RBS_SPACING_USABLE
            where = f"{site.name} at {site.start + 1}-{site.end}"
            if lo <= gap <= hi:
                out.append(
                    Issue(
                        "note",
                        f"the start codon sits {gap} nt downstream of the "
                        f"vector's {where}, inside the {lo}-{hi} nt window "
                        "where initiation works best",
                    )
                )
            elif usable_lo <= gap <= usable_hi:
                out.append(
                    Issue(
                        "warning",
                        f"the start codon sits {gap} nt from the vector's "
                        f"{where}. That is outside the {lo}-{hi} nt optimum "
                        "but still workable; expect less protein than the "
                        "vector's own cassette gave",
                    )
                )
            else:
                out.append(
                    Issue(
                        "error",
                        f"the start codon would sit {gap} nt from the "
                        f"vector's {where}, outside the {usable_lo}-"
                        f"{usable_hi} nt range a ribosome can work with. "
                        "Initiation will be poor or absent. Trim the spacer "
                        "and pad, or cut closer to the RBS",
                    )
                )

    dtm = abs(d.forward.tm_anneal - d.reverse.tm_anneal)
    if dtm > opts.max_tm_mismatch:
        out.append(
            Issue(
                "warning",
                f"annealing Tm mismatch is {dtm:.1f} C "
                f"(forward {d.forward.tm_anneal:.1f}, reverse {d.reverse.tm_anneal:.1f})",
            )
        )

    # A Tm away from target is only a problem in itself when it leaves the
    # workable band; if the two primers still match each other, the pair
    # amplifies fine at a correspondingly shifted Ta.
    for o, end in ((d.forward, "5'"), (d.reverse, "3'")):
        off = o.tm_anneal - opts.target_tm
        if abs(off) <= 5.0:
            continue
        unworkable = not 52.0 <= o.tm_anneal <= 72.0
        out.append(
            Issue(
                "warning" if unworkable else "note",
                f"{o.name}: annealing Tm {o.tm_anneal:.1f} C is "
                f"{abs(off):.1f} C {'above' if off > 0 else 'below'} the "
                f"{opts.target_tm:.0f} C target; the {end} end of the insert "
                f"is {o.gc * 100:.0f}% GC and no length in "
                f"{opts.anneal_min}-{opts.anneal_max} nt does better"
                + (
                    ". Run at the suggested Ta rather than a textbook 60 C"
                    if not unworkable
                    else ""
                ),
            )
        )

    for o in (d.forward, d.reverse):
        if not 0.35 <= o.gc <= 0.65:
            out.append(
                Issue("warning", f"{o.name}: annealing region is {o.gc * 100:.0f}% GC")
            )
        if o.longest_run >= 5:
            out.append(
                Issue("warning", f"{o.name}: run of {o.longest_run} identical bases")
            )
        if o.gc_clamp == 0:
            out.append(Issue("warning", f"{o.name}: no G/C in the last 5 bases"))
        hp = o.hairpin
        if hp.dg <= -3.0:
            out.append(
                Issue(
                    "warning" if hp.dg <= -5.0 else "note",
                    f"{o.name}: hairpin, {hp.length} bp stem {hp.seq} closed "
                    f"by a {hp.loop} nt loop, net dG {hp.dg:.1f} kcal/mol "
                    f"(stem {hp.stem_dg:.1f} + loop {hp.dg - hp.stem_dg:+.1f})"
                    + ("; it involves the 3' end" if hp.involves_3prime else ""),
                )
            )
        sd = o.self_dimer
        if sd.is_concerning:
            if sd.involves_3prime:
                tail_note = (
                    "; it reaches the 3' end, so it can be extended and will "
                    "cost yield"
                )
                level = "warning"
            elif o.dimer_is_site_palindrome:
                tail_note = (
                    "; this is the recognition site pairing with itself, which "
                    "every primer carrying a palindromic site does, and the 3' "
                    "end is free"
                )
                level = "note"
            else:
                tail_note = (
                    "; the 3' end is free, so it cannot be extended and only "
                    "sequesters primer"
                )
                level = "note"
            out.append(
                Issue(
                    level,
                    f"{o.name}: self-dimer {sd.seq} ({sd.length} bp, "
                    f"dG {sd.dg:.1f}){tail_note}",
                )
            )

    hetero = worst_duplex(d.forward.sequence, d.reverse.sequence)
    if hetero.is_concerning:
        out.append(
            Issue(
                "warning",
                f"primer pair can dimerise over {hetero.seq} "
                f"({hetero.length} bp, dG {hetero.dg:.1f})"
                + (", involving a 3' end" if hetero.involves_3prime else ""),
            )
        )

    # Sites created by the tail, as opposed to sites the gene already had.
    # Only a site lying inside the tail, or straddling the tail/annealing
    # junction, is something the design introduced; one sitting wholly in
    # the annealing region came with the template and is not news.
    for o, used in ((d.forward, d.pair.five), (d.reverse, d.pair.three)):
        tail_len = len(o.tail)
        seen: set[str] = set()
        for name, enz in ez.ENZYMES.items():
            for hit in ez.find_sites(o.sequence, enz.site):
                if hit >= tail_len:
                    continue  # inherent to the insert
                if enz.site == used.site and hit == len(o.spacer):
                    continue  # the site we put there on purpose
                if name in seen:
                    continue
                seen.add(name)
                where = (
                    "within the tail"
                    if hit + enz.length <= tail_len
                    else "across the tail/template junction"
                )
                out.append(
                    Issue(
                        "note",
                        f"{o.name}: the tail creates a {name} site {where}"
                        + (
                            " (an unavoidable consequence of the Kozak "
                            "sequence before ATG)"
                            if o.pad == "GCCACC" and name in {"NcoI", "PciI"}
                            else ""
                        ),
                    )
                )

    # Methylation blocking, checked in the real vector context.
    for enz, pos in ((d.pair.five, d.pair.pos_five), (d.pair.three, d.pair.pos_three)):
        context = d.vector.window(pos - 10, pos + enz.length + 10)
        if ez.dam_blocked(context, 10, enz):
            out.append(
                Issue(
                    "error",
                    f"{enz.name} overlaps a Dam site in the vector and will not "
                    "cut plasmid grown in a dam+ strain such as DH5-alpha; "
                    "use a dam- strain or pick another enzyme",
                )
            )
        if ez.dcm_blocked(context, 10, enz):
            out.append(
                Issue("error", f"{enz.name} is blocked by Dcm methylation here")
            )

    # Flanking bases for efficient cutting.
    # An ATG anywhere in the 5' tail is a second start codon sitting ahead
    # of the real one.  In frame it silently extends the protein's N
    # terminus; out of frame it opens a short upstream reading frame that
    # ribosomes translate instead, which can cut expression sharply.
    if d.core.startswith("ATG"):
        tail = d.forward.tail
        for hit in range(len(tail) - 2):
            if tail[hit : hit + 3] != "ATG":
                continue
            distance = len(tail) - hit
            if distance % 3 == 0:
                extra = [tail[i : i + 3] for i in range(hit, len(tail), 3)]
                out.append(
                    Issue(
                        "warning",
                        f"{d.forward.name}: the tail contains an ATG "
                        f"{distance} bases before the real start codon, in "
                        f"frame with it, so translation starting there adds "
                        f"{len(extra)} residue(s) to the N terminus "
                        f"({'-'.join(extra)})",
                    )
                )
            else:
                out.append(
                    Issue(
                        "warning",
                        f"{d.forward.name}: the tail contains an ATG "
                        f"{distance} bases before the real start codon and "
                        "out of frame with it, creating an upstream reading "
                        "frame that can suppress translation of the insert",
                    )
                )

    if d.pair.excision < opts.close_site_warning:
        out.append(
            Issue(
                "warning",
                f"the {d.pair.five.name} and {d.pair.three.name} sites are "
                f"only {d.pair.excision} bp apart in the vector; once one "
                "enzyme has cut, the other site sits that close to a fragment "
                "end and may cut poorly. Digest sequentially, or pick a pair "
                "further apart",
            )
        )

    # Spacer length only matters when the site is cut near the end of a
    # linear PCR product.  The two-step route cuts the same site out of a
    # circular plasmid instead, where it is nowhere near an end.
    for o, enz in ((d.forward, d.pair.five), (d.reverse, d.pair.three)):
        if len(o.spacer) >= enz.flank:
            continue
        if opts.two_step:
            out.append(
                Issue(
                    "note",
                    f"{o.name}: {len(o.spacer)} spacer bases is below the "
                    f"{enz.flank} the one-step route would want, but the "
                    "two-step route cuts this site out of a circular "
                    "plasmid, so it is fine here",
                )
            )
        else:
            out.append(
                Issue(
                    "warning",
                    f"{o.name}: only {len(o.spacer)} spacer bases 5' of the "
                    f"{enz.name} site; {enz.flank} is the safe minimum",
                )
            )

    # The construct must still contain exactly one of each site.
    for enz in (d.pair.five, d.pair.three):
        n = len(ez.find_sites(d.construct, enz.site, circular=True))
        if n != 1:
            out.append(
                Issue(
                    "warning",
                    f"the final construct carries {n} {enz.name} sites, "
                    "so this enzyme is no longer diagnostic on its own",
                )
            )

    # Only remark on a swapped Kozak when it really is an alternative to
    # the mammalian consensus; a yeast context is a different host's
    # sequence, not a variant of GCCACC, and in a bacterium the consensus
    # is not an option that was passed over -- it is the wrong molecule's
    # answer, so offering it as the thing that was not used is misleading.
    _variants = dict(KOZAK_VARIANTS)
    if (
        d.forward.pad in _variants
        and d.forward.pad != "GCCACC"
        and detect_host(d.vector) != "bacterial"
    ):
        out.append(
            Issue(
                "note",
                f"{d.forward.name} uses {d.forward.pad} "
                f"({_variants[d.forward.pad]}) rather than the consensus "
                "GCCACC; it keeps a purine at -3 and avoids secondary "
                "structure that the consensus would create here",
            )
        )

    if d.options.mode == "native":
        frag = d.insert_fragment
        idx = frag.find(d.core[:3])
        if d.core.startswith("ATG") and idx >= 0:
            out.append(
                Issue(
                    "note",
                    f"ATG sits {idx} bases into the ligated insert fragment",
                )
            )
    return out
