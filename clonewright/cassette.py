"""Finding the expression cassette in a bacterial plasmid that has no map.

Every decision the designer makes about a vector is made by reading feature
names: which promoter drives the insert, whether a ribosome binding site is
annotated at all, where the polylinker is, what tag the product will carry.
A vector handed over as bare FASTA has none of that, and the honest
consequences are visible -- `detect_host` returns None, the mode falls back
to `native`, and a 6xHis tag the construct really will carry goes
unmentioned.

So this module derives the cassette from the sequence instead, by looking
for the motifs that define it.  **Every search here can fail, and failing
means no feature rather than a wrong one.** That asymmetry is deliberate:
an un-annotated region produces a cautious design, while a region annotated
as something it is not produces a confident wrong one.  Nothing is inferred
from the vector's file name.

What is derivable, and what is not:

* **Derivable from sequence.** A T7 promoter (its consensus is 18 bp and
  invariant), a sigma-70 promoter (-35 and -10 boxes at a constrained
  spacing), a lac operator, a Shine-Dalgarno sequence positioned ahead of a
  start codon, a run of six histidines in frame, a protease recognition
  site read as peptide, a transcriptional terminator's inverted repeat, the
  polylinker as the cluster where unique cutters crowd together, and open
  reading frames.
* **Not derivable.** What an open reading frame *is*.  A 795 bp ORF is a
  795 bp ORF; that it confers kanamycin resistance is knowledge about the
  vector, not a property of its bases.  Names for those come from
  `hints`, supplied by whoever builds the map, and an ORF with no hint is
  annotated by its length and left unnamed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import enzymes as ez
from .seqio import Feature, Record, revcomp, translate

# The T7 promoter's class III consensus.  Phage T7 RNA polymerase needs
# essentially all of it, which is why this is a motif search and not a
# scored alignment: a T7 promoter is either there or it is not.
T7_CONSENSUS = "TAATACGACTCACTATAG"

# The lac operator, O1.  Matched with a tolerance because vectors carry
# variants of it -- pET's is not quite pUC's -- and because a lac operator
# is weak evidence anyway: it says the cassette is repressible, nothing
# more, and the design never turns on it.
LAC_OPERATOR = "GGAATTGTGAGCGGATAACAATT"
LAC_OPERATOR_TOLERANCE = 4

# Sigma-70's two boxes, and the spacing between them.  The spacing is the
# discriminating part: the hexamers are common enough to occur by chance
# every few hundred bases, and a pair of them at 15-19 bp is not.
MINUS_35 = "TTGACA"
MINUS_10 = "TATAAT"
SPACER_RANGE = (15, 19)

# How much of each box has to match.  Five of six in both, which admits
# tac and trc (hybrids that are consensus at one box and near it at the
# other) and rejects most of the chance hexamer pairs.
BOX_MIN_MATCH = 5

# Protease sites, as the peptide the protease reads.  Searching the
# translation rather than the DNA is what makes these robust: the same
# recognition site is encoded differently in different vectors, and a DNA
# motif for one of them finds nothing in the next.
PROTEASE_SITES: dict[str, str] = {
    "thrombin site": "LVPRGS",
    "TEV site": "ENLYFQ",
    "factor Xa site": "IEGR",
    "enterokinase site": "DDDDK",
    "PreScission site": "LEVLFQGP",
}

# Affinity and epitope tags, likewise as peptide.
PEPTIDE_TAGS: dict[str, str] = {
    "6xHis tag": "HHHHHH",
    "S-tag": "KETAAAKFERQHMDS",
    "FLAG tag": "DYKDDDDK",
    "Myc tag": "EQKLISEEDL",
    "HA tag": "YPYDVPDYA",
    "Strep-tag II": "WSHPQFEK",
}

# A Shine-Dalgarno sequence is a purine-rich stretch complementary to the
# 3' end of 16S rRNA.  Full consensus is AGGAGG; real vectors carry
# shortened and shifted versions, so a 6-mer window scoring at least four
# matches against the consensus counts, and the spacing test is what
# decides whether it is the one driving the start codon.
SHINE_DALGARNO = "AGGAGG"
SD_MIN_MATCH = 4
# How far ahead of a start codon the ribosome binding site can sit and
# still be the one that initiates there.  Wider than the optimum, because
# the job here is to find the site, not to judge the spacing -- that
# judgement belongs to `design.rbs_spacing`, on the finished construct.
SD_WINDOW = (3, 20)


def _matches(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x == y)


@dataclass(frozen=True)
class Hints:
    """What cannot be read off the sequence.

    `orfs` maps a name to the length in base pairs of the coding sequence
    it names, and an open reading frame is given that name when its length
    is within `orf_tolerance` of it.  Lengths are published properties of
    these vectors; matching on them is arithmetic.
    """

    promoter: str = ""
    partner: str = ""
    orfs: dict[str, int] = field(default_factory=dict)
    orf_tolerance: float = 0.05


def t7_promoters(seq: str, *, circular: bool = False) -> list[Feature]:
    """Every T7 promoter, on both strands."""
    out: list[Feature] = []
    search = seq + seq[: len(T7_CONSENSUS) - 1] if circular else seq
    for forward in (True, False):
        motif = T7_CONSENSUS if forward else revcomp(T7_CONSENSUS)
        for m in re.finditer(f"(?={re.escape(motif)})", search):
            start = m.start() % len(seq)
            out.append(Feature(
                "T7 promoter", "promoter", start, start + len(motif), forward
            ))
    return out


def sigma70_promoters(seq: str, *, name: str = "promoter") -> list[Feature]:
    """Sigma-70 promoters, by the geometry of their two boxes.

    Reported from the start of the -35 box to the end of the -10 box, on
    the strand transcription would run along.  Overlapping candidates are
    reduced to the best-scoring one, because the boxes tolerate mismatch
    and a real promoter otherwise comes back three times at one-base
    offsets.
    """
    found: list[tuple[int, Feature]] = []
    for forward in (True, False):
        strand = seq if forward else revcomp(seq)
        n = len(strand)
        for i in range(n - 6):
            box35 = strand[i : i + 6]
            score35 = _matches(box35, MINUS_35)
            if score35 < BOX_MIN_MATCH:
                continue
            for gap in range(SPACER_RANGE[0], SPACER_RANGE[1] + 1):
                j = i + 6 + gap
                if j + 6 > n:
                    break
                score10 = _matches(strand[j : j + 6], MINUS_10)
                if score10 < BOX_MIN_MATCH:
                    continue
                start, end = i, j + 6
                if not forward:
                    start, end = n - (j + 6), n - i
                found.append((
                    score35 + score10,
                    Feature(name, "promoter", start, end, forward),
                ))
    # Keep the best of each overlapping group.
    out: list[Feature] = []
    for score, f in sorted(found, key=lambda pair: -pair[0]):
        if any(f.start < g.end and g.start < f.end and f.forward == g.forward
               for g in out):
            continue
        out.append(f)
    return sorted(out, key=lambda f: f.start)


def operators(seq: str) -> list[Feature]:
    """lac operators, within a few mismatches of O1."""
    out: list[Feature] = []
    width = len(LAC_OPERATOR)
    for forward in (True, False):
        motif = LAC_OPERATOR if forward else revcomp(LAC_OPERATOR)
        for i in range(len(seq) - width + 1):
            if _matches(seq[i : i + width], motif) >= width - LAC_OPERATOR_TOLERANCE:
                out.append(Feature(
                    "lac operator", "protein_bind", i, i + width, forward
                ))
    # Collapse neighbours arising from the mismatch tolerance.
    collapsed: list[Feature] = []
    for f in out:
        if collapsed and f.start - collapsed[-1].start < width:
            continue
        collapsed.append(f)
    return collapsed


def _orfs_on(strand: str, min_bp: int) -> list[tuple[int, int]]:
    """Start-to-stop spans on one strand, longest first, non-overlapping."""
    spans: list[tuple[int, int]] = []
    for frame in range(3):
        i = frame
        while i + 3 <= len(strand):
            if strand[i : i + 3] != "ATG":
                i += 3
                continue
            j = i
            while j + 3 <= len(strand):
                codon = strand[j : j + 3]
                if codon in ("TAA", "TAG", "TGA"):
                    if j + 3 - i >= min_bp:
                        spans.append((i, j + 3))
                    i = j + 3
                    break
                j += 3
            else:
                i = j
                break
            continue
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)
    kept: list[tuple[int, int]] = []
    for start, end in spans:
        if any(start < k_end and k_start < end for k_start, k_end in kept):
            continue
        kept.append((start, end))
    return kept


# How much of a shorter open reading frame may lie inside a longer one on
# the other strand before it is treated as noise.
#
# This bound is not cosmetic, and the case that produced it is worth
# keeping.  The reverse strand of pGEX-4T-1's GST gene carries a 111-codon
# ORF by chance -- a real arrangement of bases, and not a gene.  Annotated,
# it became a feature overlapping the start codon, and `FusionFrame`
# counts any such overlap as *conflicting* evidence about the frame; one
# conflict makes the frame unconfirmed, `choose_mode` then finds no leader
# to fuse behind, and pGEX-4T-1 -- a vector whose entire purpose is an
# N-terminal GST fusion -- came back as `native`.  The design was still
# sound and silently answered a different question than the one asked.
#
# A chance ORF antisense to a real coding sequence is common; a genuine
# gene nested inside another on the opposite strand is rare in a cloning
# vector.  So the overlap loses.
ANTISENSE_OVERLAP_LIMIT = 0.5


def open_reading_frames(
    seq: str, *, min_bp: int = 300, hints: Hints | None = None
) -> list[Feature]:
    """Open reading frames, named from `hints` where a length matches.

    An ORF with no matching hint keeps its length as its name.  That reads
    oddly in a feature table and is the point: it says what was measured
    and does not say what it is.

    Chance reading frames antisense to a longer one are dropped -- see
    `ANTISENSE_OVERLAP_LIMIT`, which exists because one of them turned
    pGEX-4T-1's GST fusion into a native clone.
    """
    hints = hints or Hints()
    n = len(seq)
    found: list[Feature] = []
    for forward in (True, False):
        strand = seq if forward else revcomp(seq)
        for start, end in _orfs_on(strand, min_bp):
            if forward:
                lo, hi = start, end
            else:
                lo, hi = n - end, n - start
            length = end - start
            name = ""
            best = None
            for candidate, expected in hints.orfs.items():
                drift = abs(length - expected) / expected
                if drift <= hints.orf_tolerance and (best is None or drift < best):
                    best, name = drift, candidate
            found.append(Feature(
                name or f"unnamed ORF ({length // 3 - 1} aa)",
                "CDS", lo, hi, forward,
            ))

    out: list[Feature] = []
    for f in sorted(found, key=len, reverse=True):
        shadowed = False
        for g in out:
            if g.forward == f.forward or len(g) <= len(f):
                continue
            overlap = min(f.end, g.end) - max(f.start, g.start)
            if overlap > 0 and overlap / len(f) > ANTISENSE_OVERLAP_LIMIT:
                shadowed = True
                break
        if not shadowed:
            out.append(f)
    return sorted(out, key=lambda f: (f.start, f.end))


def peptide_features(
    seq: str, *, table: dict[str, str], kind: str, circular: bool = False
) -> list[Feature]:
    """Features found by translating, not by matching DNA.

    Six frames, and a hit is converted back to the nucleotide span it was
    translated from.  This is how a tag or a protease site is found in a
    vector whose codon usage nobody wrote down.
    """
    out: list[Feature] = []
    n = len(seq)
    for forward in (True, False):
        strand = seq if forward else revcomp(seq)
        for frame in range(3):
            protein = translate(strand[frame:])
            for name, peptide in table.items():
                for m in re.finditer(f"(?={re.escape(peptide)})", protein):
                    start = frame + m.start() * 3
                    end = start + len(peptide) * 3
                    if forward:
                        lo, hi = start, end
                    else:
                        lo, hi = n - end, n - start
                    out.append(Feature(name, kind, lo, hi, forward))
    return sorted(out, key=lambda f: f.start)


def ribosome_binding_sites(
    seq: str, starts: list[tuple[int, bool]]
) -> list[Feature]:
    """Shine-Dalgarno sequences sitting the right distance ahead of a start.

    `starts` is the start codons worth initiating at, as (position, strand).
    A purine-rich window is only called a ribosome binding site when one of
    them is in range: the sequence alone occurs too often to mean anything,
    and what makes it an RBS is where it sits.
    """
    n = len(seq)
    width = len(SHINE_DALGARNO)
    out: list[Feature] = []
    seen: set[tuple[int, int, bool]] = set()
    for atg, forward in starts:
        strand = seq if forward else revcomp(seq)
        pos = atg if forward else n - atg - 3
        best: tuple[int, int] | None = None
        for gap in range(SD_WINDOW[0], SD_WINDOW[1] + 1):
            end = pos - gap
            start = end - width
            if start < 0:
                continue
            score = _matches(strand[start:end], SHINE_DALGARNO)
            if score < SD_MIN_MATCH:
                continue
            if best is None or score > best[0]:
                best = (score, start)
        if best is None:
            continue
        start = best[1]
        if forward:
            lo, hi = start, start + width
        else:
            lo, hi = n - start - width, n - start
        key = (lo, hi, forward)
        if key in seen:
            continue
        seen.add(key)
        out.append(Feature("RBS", "RBS", lo, hi, forward))
    return out


def inverted_repeats(
    seq: str, *, min_stem: int = 7, max_loop: int = 12, window: tuple[int, int] | None = None
) -> list[Feature]:
    """Hairpins strong enough to be transcriptional terminators.

    A terminator is a structure, not a sequence, so this looks for the
    structure: a stem of at least `min_stem` perfectly paired bases closing
    a short loop.  It is named `terminator` only in the sense that this is
    what one looks like -- which phage or gene it came from is not
    something the bases say.
    """
    lo, hi = window or (0, len(seq))
    out: list[Feature] = []
    region = seq[lo:hi]
    i = 0
    while i < len(region) - 2 * min_stem:
        hit = None
        for stem in range(min_stem + 6, min_stem - 1, -1):
            left = region[i : i + stem]
            if left.count("G") + left.count("C") < stem * 0.6:
                continue
            for loop in range(3, max_loop + 1):
                j = i + stem + loop
                if j + stem > len(region):
                    break
                if region[j : j + stem] == revcomp(left):
                    hit = (stem, loop, j + stem)
                    break
            if hit:
                break
        if hit:
            stem, loop, end = hit
            out.append(Feature(
                "terminator", "terminator", lo + i, lo + end, True
            ))
            i = end
        else:
            i += 1
    return out


def polylinker(record: Record) -> Feature | None:
    """The polylinker, as the stretch where unique cutters crowd together.

    A multiple cloning site is not a sequence either.  What it is, is a
    short run of the plasmid carrying several sites that each occur exactly
    once in it -- which is a measurable property, and the one the designer
    actually needs.
    """
    positions: list[int] = []
    for name, enzyme in ez.ENZYMES.items():
        sites = ez.find_sites(record.seq, enzyme.site, circular=record.circular)
        if len(sites) == 1:
            positions.append(sites[0])
    if len(positions) < 3:
        return None
    positions.sort()
    best: tuple[int, int, int] | None = None
    for i, start in enumerate(positions):
        j = i
        while j + 1 < len(positions) and positions[j + 1] - start <= 120:
            j += 1
        count = j - i + 1
        if best is None or count > best[0]:
            best = (count, start, positions[j])
    if best is None or best[0] < 3:
        return None
    _, start, end = best
    return Feature("MCS", "misc_feature", start, end + 6, True)


def annotate(
    record: Record, *, hints: Hints | None = None, min_orf_bp: int = 300
) -> Record:
    """A new Record carrying every cassette feature the sequence supports.

    The input is not modified and its own features, if it had any, are kept
    and come first: a derived annotation is a weaker claim than one someone
    made deliberately, and overwriting the latter with the former would be
    the wrong way round.
    """
    hints = hints or Hints()
    seq = record.seq
    features: list[Feature] = list(record.features)

    promoters = t7_promoters(seq, circular=record.circular)
    if not promoters:
        promoters = sigma70_promoters(seq, name=hints.promoter or "promoter")
    features += promoters
    features += operators(seq)

    orfs = open_reading_frames(seq, min_bp=min_orf_bp, hints=hints)
    features += orfs

    tags = peptide_features(
        seq, table=PEPTIDE_TAGS, kind="CDS", circular=record.circular
    )
    features += tags
    features += peptide_features(
        seq, table=PROTEASE_SITES, kind="CDS", circular=record.circular
    )

    # Start codons worth testing for a ribosome binding site.  Two ways in,
    # because vectors are built both ways: pET-28a opens with a 6xHis tag
    # and no fusion partner, while pGEX and pMAL open with a 200-to-400
    # residue partner and carry no peptide tag at all.  Seeding only from
    # tags found nothing in the second kind, and a vector with no derived
    # RBS reads as a non-bacterial one two steps later.
    starts: list[tuple[int, bool]] = []

    for tag in tags:
        # The initiator the tag sits in frame with: walk upstream in
        # threes to the first ATG.
        if tag.forward:
            upstream = max(tag.start - 90, 0)
            region = seq[upstream : tag.start]
            offset = (tag.start - upstream) % 3
            for i in range(offset, len(region) - 2, 3):
                if region[i : i + 3] == "ATG":
                    starts.append((upstream + i, True))
                    break
        else:
            downstream = min(tag.end + 90, len(seq))
            region = revcomp(seq[tag.end : downstream])
            offset = (downstream - tag.end) % 3
            for i in range(offset, len(region) - 2, 3):
                if region[i : i + 3] == "ATG":
                    starts.append((downstream - i - 3, False))
                    break

    # An open reading frame that begins just after a promoter, on the
    # strand that promoter transcribes: that is a cassette's own start
    # codon, whatever the gene in front of it turns out to be.
    for promoter in promoters:
        for orf in orfs:
            if orf.forward != promoter.forward:
                continue
            if promoter.forward and -10 <= orf.start - promoter.end <= 200:
                starts.append((orf.start, True))
            elif not promoter.forward and -10 <= promoter.start - orf.end <= 200:
                starts.append((orf.end - 3, False))

    sites = ribosome_binding_sites(seq, starts)
    features += sites

    mcs = polylinker(record)
    if mcs is not None and sites:
        # Clip the polylinker to the translated side of the ribosome
        # binding site.  `polylinker` finds the stretch where unique
        # cutters crowd together, and on the pET vectors that cluster
        # reaches back past the RBS to pick up an XbaI site which is part
        # of the upstream region, not part of the cloning site.
        #
        # Left as it was, pET-21b -- a vector whose whole cloning strategy
        # is "bring your own ATG in an NdeI site" -- offered that XbaI as
        # its 5' enzyme, which puts the start codon 29 nt *upstream* of the
        # RBS. The designer was right to refuse it; the annotation was
        # wrong to offer it. A cloning site cannot precede the sequence
        # that positions the ribosome on it.
        nearest = min(sites, key=lambda f: min(
            abs(f.start - mcs.end), abs(mcs.start - f.end)
        ))
        if nearest.forward:
            mcs = Feature(mcs.name, mcs.kind, max(mcs.start, nearest.end),
                          mcs.end, mcs.forward)
        else:
            mcs = Feature(mcs.name, mcs.kind, mcs.start,
                          min(mcs.end, nearest.start), mcs.forward)
        if len(mcs) < 12:
            mcs = None

    if mcs is not None:
        features += [mcs]
        # A terminator belongs just past the end of the cassette, so only
        # the flanks of the polylinker are searched: inverted repeats occur
        # all over a plasmid, and annotating every one of them as a
        # terminator would be a guess dressed up as a finding.
        #
        # Both flanks, because which one is "past the end" depends on which
        # strand the cassette reads along, and that is not known yet --
        # pET-28a's runs against the map, putting its terminator at a lower
        # coordinate than its polylinker.
        for lo, hi in (
            (mcs.end, min(mcs.end + 400, len(seq))),
            (max(mcs.start - 400, 0), mcs.start),
        ):
            features += inverted_repeats(seq, window=(lo, hi))[:1]

    if hints.partner:
        # The N-terminal fusion partner: the longest ORF that reaches the
        # polylinker from upstream, which is a geometric test.  Its name is
        # a hint because no test on the bases yields it.
        anchor = mcs.start if mcs is not None else None
        if anchor is not None:
            reaching = [
                f for f in orfs
                if f.start < anchor and anchor - f.end < 120
            ]
            if reaching:
                chosen = max(reaching, key=len)
                features = [
                    Feature(hints.partner, "CDS", chosen.start, chosen.end,
                            chosen.forward)
                    if f is chosen else f
                    for f in features
                ]

    deduped: list[Feature] = []
    for f in sorted(features, key=lambda f: (f.start, f.end, f.name)):
        if any(g.name == f.name and g.start == f.start and g.end == f.end
               for g in deduped):
            continue
        deduped.append(f)
    return Record(record.name, seq, record.circular, deduped)
