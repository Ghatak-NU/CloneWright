"""Type II restriction enzymes, site finding, and methylation blocking.

Cut positions are offsets from the first base of the recognition site.
`top` is where the top strand is cut, `bottom` where the bottom strand is,
both counted in top-strand coordinates.  So for EcoRI, G^AATTC, top=1 and
bottom=5, and the difference gives a 4 nt 5' overhang.  A negative
difference is a 3' overhang; zero is blunt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .seqio import IUPAC, revcomp

# Bases to place 5' of the site so the enzyme has something to hold on to.
# NEB publishes per-enzyme figures; 6 is a conservative value that is at or
# above the requirement for every enzyme in this table, so it is used unless
# an enzyme overrides it.  Check NEB's "cleavage close to the end of DNA
# fragments" table if you want to trim a primer further.
DEFAULT_FLANK = 6


@dataclass(frozen=True)
class Enzyme:
    name: str
    site: str
    top: int
    bottom: int
    flank: int = DEFAULT_FLANK
    dam_sensitive: bool = False
    dcm_sensitive: bool = False
    rank: int = 50  # lower is preferred when two enzymes are interchangeable

    @property
    def length(self) -> int:
        return len(self.site)

    @property
    def overhang(self) -> int:
        """>0 is a 5' overhang, <0 a 3' overhang, 0 blunt."""
        return self.bottom - self.top

    @property
    def is_blunt(self) -> bool:
        return self.overhang == 0

    @property
    def is_degenerate(self) -> bool:
        return any(b not in "ACGT" for b in self.site)

    @property
    def overhang_seq(self) -> str:
        """The sticky end produced, as written on the top strand."""
        if self.overhang > 0:
            return self.site[self.top : self.bottom]
        if self.overhang < 0:
            return self.site[self.bottom : self.top]
        return ""

    def describe(self) -> str:
        cut = f"{self.site[: self.top]}^{self.site[self.top :]}"
        if self.is_blunt:
            end = "blunt"
        else:
            end = f"{abs(self.overhang)} nt {'5' if self.overhang > 0 else '3'}' overhang"
        return f"{self.name} {cut} ({end})"


def _e(name, site, top, bottom, **kw) -> tuple[str, Enzyme]:
    return name, Enzyme(name, site, top, bottom, **kw)


ENZYMES: dict[str, Enzyme] = dict(
    [
        _e("AatII", "GACGTC", 5, 1),
        _e("AccI", "GTMKAC", 2, 4),
        _e("AflII", "CTTAAG", 1, 5),
        _e("AflIII", "ACRYGT", 1, 5),
        _e("AgeI", "ACCGGT", 1, 5),
        _e("ApaI", "GGGCCC", 5, 1),
        _e("AscI", "GGCGCGCC", 2, 6),
        _e("AseI", "ATTAAT", 2, 4),
        _e("AvrII", "CCTAGG", 1, 5),
        _e("BamHI", "GGATCC", 1, 5),
        _e("BclI", "TGATCA", 1, 5, dam_sensitive=True),
        _e("BglII", "AGATCT", 1, 5),
        _e("BmtI", "GCTAGC", 5, 1),
        _e("BsiWI", "CGTACG", 1, 5),
        _e("BspEI", "TCCGGA", 1, 5),
        _e("BspHI", "TCATGA", 1, 5, dam_sensitive=True),
        _e("BsrGI", "TGTACA", 1, 5),
        _e("BssHII", "GCGCGC", 1, 5),
        _e("BstBI", "TTCGAA", 2, 4),
        _e("ClaI", "ATCGAT", 2, 4, dam_sensitive=True),
        _e("DraI", "TTTAAA", 3, 3),
        _e("EagI", "CGGCCG", 1, 5),
        _e("EcoRI", "GAATTC", 1, 5),
        _e("EcoRV", "GATATC", 3, 3),
        _e("FseI", "GGCCGGCC", 6, 2),
        _e("HindIII", "AAGCTT", 1, 5),
        _e("HpaI", "GTTAAC", 3, 3),
        _e("KasI", "GGCGCC", 1, 5),
        _e("KpnI", "GGTACC", 5, 1),
        _e("MfeI", "CAATTG", 1, 5),
        _e("MluI", "ACGCGT", 1, 5),
        _e("MscI", "TGGCCA", 3, 3),
        _e("NarI", "GGCGCC", 2, 4),
        _e("NcoI", "CCATGG", 1, 5),
        _e("NdeI", "CATATG", 2, 4),
        _e("NheI", "GCTAGC", 1, 5),
        _e("NotI", "GCGGCCGC", 2, 6),
        _e("NruI", "TCGCGA", 3, 3),
        _e("NsiI", "ATGCAT", 5, 1),
        _e("PacI", "TTAATTAA", 5, 3),
        _e("PciI", "ACATGT", 1, 5),
        _e("PmeI", "GTTTAAAC", 4, 4),
        _e("PmlI", "CACGTG", 3, 3),
        _e("PspOMI", "GGGCCC", 1, 5),
        _e("PstI", "CTGCAG", 5, 1),
        _e("PvuI", "CGATCG", 4, 2),
        _e("PvuII", "CAGCTG", 3, 3),
        _e("SacI", "GAGCTC", 5, 1),
        _e("SacII", "CCGCGG", 4, 2),
        _e("SalI", "GTCGAC", 1, 5),
        _e("SbfI", "CCTGCAGG", 6, 2),
        _e("ScaI", "AGTACT", 3, 3),
        _e("SmaI", "CCCGGG", 3, 3),
        _e("SpeI", "ACTAGT", 1, 5),
        _e("SphI", "GCATGC", 5, 1),
        _e("SspI", "AATATT", 3, 3),
        _e("StuI", "AGGCCT", 3, 3),
        _e("SwaI", "ATTTAAAT", 4, 4),
        _e("XbaI", "TCTAGA", 1, 5, dam_sensitive=True),
        _e("XhoI", "CTCGAG", 1, 5),
        _e("XmaI", "CCCGGG", 1, 5),
    ]
)

# Enzymes most labs keep in the freezer.  Several entries above are
# isoschizomers that differ only in where they cut inside a shared site
# (NheI/BmtI, ApaI/PspOMI, KasI/NarI, SmaI/XmaI); ranking lets the common
# one win so the designer does not hand back an obscure equivalent.
_WORKHORSES = {
    "AflII", "AgeI", "ApaI", "AscI", "AvrII", "BamHI", "BglII", "BsiWI",
    "BspEI", "BsrGI", "ClaI", "DraI", "EcoRI", "EcoRV", "FseI", "HindIII",
    "HpaI", "KpnI", "MluI", "NcoI", "NdeI", "NheI", "NotI", "PacI", "PmeI",
    "PstI", "PvuII", "SacI", "SalI", "SbfI", "ScaI", "SmaI", "SpeI", "SphI",
    "SspI", "StuI", "XbaI", "XhoI",
}
for _name in _WORKHORSES:
    _enz = ENZYMES[_name]
    ENZYMES[_name] = Enzyme(
        _enz.name, _enz.site, _enz.top, _enz.bottom, _enz.flank,
        _enz.dam_sensitive, _enz.dcm_sensitive, rank=10,
    )
del _name, _enz

# ---------------------------------------------------------------------------
# reaction conditions
# ---------------------------------------------------------------------------

# The four standard buffers, ordered by ionic strength.  A sequential
# digest goes low salt first and then has salt added, because salt can be
# raised in a tube but not lowered.
BUFFERS: tuple[str, ...] = ("r1.1", "r2.1", "rCutSmart", "r3.1")


@dataclass(frozen=True)
class Reaction:
    """Incubation conditions for one enzyme.

    `activity` is percent activity in each of `BUFFERS`, in that order.
    `heat_kill` is (temperature, minutes), or None meaning *not recorded
    here* -- which is treated as "cannot be heat killed", so the protocol
    falls back to a column.  That asymmetry is deliberate: a missing entry
    costs a spin column, whereas assuming an enzyme dies at 65 C when it
    does not leaves it active through the ligation.

    These figures are transcribed from the suppliers' published activity
    charts.  They are the one part of this tool that is neither computed
    nor derived from the sequences, so they can go stale when a supplier
    reformulates, and they are printed in the protocol rather than only
    used, so a wrong one is visible rather than silent.  Check the current
    chart before setting up a digest that matters.
    """

    temp_c: int = 37
    activity: tuple[int, int, int, int] = (0, 0, 0, 0)
    heat_kill: tuple[int, int] | None = None
    star_prone: bool = False

    def percent(self, buffer: str) -> int:
        try:
            return self.activity[BUFFERS.index(buffer)]
        except ValueError:
            return 0

    @property
    def is_recorded(self) -> bool:
        return any(self.activity)


UNKNOWN_REACTION = Reaction()

# name -> (temp, (r1.1, r2.1, rCutSmart, r3.1), heat kill, star-prone)
_R: dict[str, Reaction] = {
    "AatII": Reaction(37, (10, 50, 100, 10), (80, 20)),
    "AccI": Reaction(37, (25, 100, 100, 75), (65, 20)),
    "AflII": Reaction(37, (10, 50, 100, 100), (65, 20)),
    "AflIII": Reaction(37, (10, 75, 50, 100), (80, 20)),
    "AgeI": Reaction(37, (100, 75, 100, 10), (65, 20)),
    "ApaI": Reaction(25, (10, 25, 100, 100), (65, 20)),
    "AscI": Reaction(37, (10, 50, 100, 25), (65, 20)),
    "AseI": Reaction(37, (50, 100, 75, 100), (65, 20)),
    "AvrII": Reaction(37, (10, 50, 100, 100), None),
    "BamHI": Reaction(37, (25, 100, 100, 100), (80, 20), star_prone=True),
    "BclI": Reaction(50, (25, 100, 75, 100), None),
    "BglII": Reaction(37, (10, 50, 100, 100), (80, 20)),
    "BmtI": Reaction(37, (10, 100, 100, 50), (80, 20)),
    "BsiWI": Reaction(55, (10, 100, 50, 50), (80, 20)),
    "BspEI": Reaction(37, (10, 50, 100, 100), (80, 20)),
    "BspHI": Reaction(37, (10, 50, 75, 100), (80, 20)),
    "BsrGI": Reaction(37, (10, 100, 100, 75), (80, 20)),
    "BssHII": Reaction(50, (10, 50, 50, 100), (65, 20)),
    "BstBI": Reaction(65, (25, 75, 100, 100), None),
    "ClaI": Reaction(37, (50, 75, 100, 50), (65, 20)),
    "DraI": Reaction(37, (25, 100, 100, 75), (65, 20)),
    "EagI": Reaction(37, (10, 25, 100, 100), (65, 20)),
    "EcoRI": Reaction(37, (25, 100, 100, 50), (65, 20), star_prone=True),
    "EcoRV": Reaction(37, (25, 100, 100, 100), (80, 20), star_prone=True),
    "FseI": Reaction(37, (10, 25, 100, 50), (65, 20)),
    "HindIII": Reaction(37, (10, 100, 100, 50), (80, 20), star_prone=True),
    "HpaI": Reaction(37, (50, 100, 100, 10), (65, 20)),
    "KasI": Reaction(37, (10, 25, 100, 50), (65, 20)),
    "KpnI": Reaction(37, (100, 50, 100, 10), None),
    "MfeI": Reaction(37, (10, 75, 100, 100), (65, 20)),
    "MluI": Reaction(37, (10, 50, 100, 100), (80, 20)),
    "MscI": Reaction(37, (10, 50, 100, 25), (65, 20)),
    "NarI": Reaction(37, (25, 50, 100, 25), None),
    "NcoI": Reaction(37, (10, 50, 100, 100), (80, 20)),
    "NdeI": Reaction(37, (50, 100, 100, 75), (65, 20)),
    "NheI": Reaction(37, (10, 100, 100, 50), (80, 20)),
    "NotI": Reaction(37, (10, 25, 100, 100), (65, 20)),
    "NruI": Reaction(37, (25, 50, 100, 25), (80, 20)),
    "NsiI": Reaction(37, (25, 75, 100, 100), (80, 20)),
    "PacI": Reaction(37, (10, 25, 100, 50), (65, 20)),
    "PciI": Reaction(37, (10, 75, 50, 100), None),
    "PmeI": Reaction(37, (25, 50, 100, 75), (65, 20)),
    "PmlI": Reaction(37, (10, 50, 100, 100), (65, 20)),
    "PspOMI": Reaction(37, (10, 50, 75, 100), (65, 20)),
    "PstI": Reaction(37, (25, 75, 100, 100), (80, 20), star_prone=True),
    "PvuI": Reaction(37, (10, 50, 100, 100), (80, 20)),
    "PvuII": Reaction(37, (50, 100, 100, 25), (65, 20), star_prone=True),
    "SacI": Reaction(37, (100, 50, 100, 10), (65, 20)),
    "SacII": Reaction(37, (10, 50, 100, 25), (65, 20)),
    "SalI": Reaction(37, (10, 50, 100, 100), (65, 20), star_prone=True),
    "SbfI": Reaction(37, (25, 50, 100, 100), (80, 20)),
    "ScaI": Reaction(37, (25, 75, 100, 100), (80, 20)),
    "SmaI": Reaction(25, (25, 50, 100, 10), (65, 20)),
    "SpeI": Reaction(37, (10, 100, 100, 75), (80, 20)),
    "SphI": Reaction(37, (10, 100, 100, 50), (65, 20)),
    "SspI": Reaction(37, (50, 100, 100, 50), (65, 20)),
    "StuI": Reaction(37, (25, 75, 100, 50), (65, 20)),
    "SwaI": Reaction(25, (50, 100, 75, 25), (65, 20)),
    "XbaI": Reaction(37, (25, 75, 100, 100), (65, 20)),
    "XhoI": Reaction(37, (25, 75, 100, 100), (65, 20)),
    "XmaI": Reaction(37, (50, 100, 75, 10), (65, 20)),
}


def reaction(enzyme: Enzyme | str) -> Reaction:
    name = enzyme if isinstance(enzyme, str) else enzyme.name
    return _R.get(name, UNKNOWN_REACTION)


@dataclass(frozen=True)
class DigestPlan:
    """How to cut with two enzymes: together, or one after the other."""

    first: Enzyme
    second: Enzyme
    simultaneous: bool
    buffer: str | None  # the shared buffer, when there is one
    temp_c: int | None  # the shared temperature, when there is one
    activities: tuple[int, int]  # of first, second, in `buffer`
    reason: str
    heat_kill: tuple[int, int] | None  # kills both, when such a step exists
    notes: tuple[str, ...] = ()

    @property
    def order(self) -> tuple[Enzyme, Enzyme]:
        return self.first, self.second


def plan_digest(
    a: Enzyme, b: Enzyme, *, min_activity: int = 75
) -> DigestPlan:
    """Decide whether `a` and `b` can go in one tube, and in what.

    Two things force a sequential digest: incubation temperatures that do
    not match, and no buffer in which both enzymes work well enough.  The
    threshold is deliberately set above half activity -- an enzyme at 25%
    will usually still finish a digest given time and units, but planning
    for it means planning for a partial, and a partially cut vector is
    the single commonest reason a ligation gives nothing but background.

    When it has to be sequential the low-salt enzyme goes first, because
    salt can be added to a tube and not taken out.
    """
    ra, rb = reaction(a), reaction(b)
    notes: list[str] = []

    for enzyme, r in ((a, ra), (b, rb)):
        if not r.is_recorded:
            return DigestPlan(
                a, b, False, None, None, (0, 0),
                f"no reaction conditions recorded for {enzyme.name}",
                None,
                ("look the enzyme up and digest sequentially, which is "
                 "always safe, or use the supplier's double-digest finder",),
            )
        if r.star_prone:
            notes.append(
                f"{enzyme.name} shows star activity when over-digested: keep "
                "glycerol under 5% of the reaction and do not run it "
                "overnight"
            )

    if ra.temp_c != rb.temp_c:
        low, high = (a, b) if ra.temp_c < rb.temp_c else (b, a)
        return DigestPlan(
            low, high, False, None, None, (0, 0),
            f"{a.name} runs at {ra.temp_c} C and {b.name} at {rb.temp_c} C",
            None,
            tuple(notes)
            + (
                f"digest at {min(ra.temp_c, rb.temp_c)} C with "
                f"{low.name} first, then raise to "
                f"{max(ra.temp_c, rb.temp_c)} C and add {high.name}",
            ),
        )

    shared = [
        (min(ra.percent(buf), rb.percent(buf)), buf)
        for buf in BUFFERS
        if min(ra.percent(buf), rb.percent(buf)) >= min_activity
    ]
    if shared:
        # Best for the weaker enzyme; BUFFERS is ordered so that the
        # gentler buffer wins a tie.
        best, buf = max(shared, key=lambda row: (row[0], -BUFFERS.index(row[1])))
        kill = None
        if ra.heat_kill and rb.heat_kill:
            kill = (
                max(ra.heat_kill[0], rb.heat_kill[0]),
                max(ra.heat_kill[1], rb.heat_kill[1]),
            )
        else:
            dead = [
                e.name for e, r in ((a, ra), (b, rb)) if r.heat_kill is None
            ]
            notes.append(
                f"no heat inactivation recorded for {', '.join(dead)}, so "
                "clean up on a column rather than relying on a heat step"
            )
        return DigestPlan(
            a, b, True, buf, ra.temp_c,
            (ra.percent(buf), rb.percent(buf)),
            f"both enzymes are at least {best}% active in {buf}",
            kill, tuple(notes),
        )

    salt_order = {name: i for i, name in enumerate(BUFFERS)}

    def gentlest(r: Reaction) -> int:
        best = [buf for buf in BUFFERS if r.percent(buf) >= min_activity]
        return salt_order[best[0]] if best else len(BUFFERS)

    first, second = (a, b) if gentlest(ra) <= gentlest(rb) else (b, a)
    return DigestPlan(
        first, second, False, None, ra.temp_c, (0, 0),
        f"no single buffer leaves both {a.name} and {b.name} at "
        f"{min_activity}% or more",
        None,
        tuple(notes)
        + (
            f"digest with {first.name} in its own buffer first, then bring "
            f"the salt up and add {second.name}; salt can be added to a tube "
            "and not removed",
        ),
    )


DAM_SITE = "GATC"
DCM_SITE = "CCWGG"


def isoschizomers(enzyme: Enzyme) -> list[str]:
    """Other enzymes in the table sharing this recognition sequence."""
    return sorted(
        n for n, e in ENZYMES.items()
        if e.site == enzyme.site and n != enzyme.name
    )


class EnzymeError(KeyError):
    pass


def get(name: str) -> Enzyme:
    for key, enz in ENZYMES.items():
        if key.lower() == name.strip().lower():
            return enz
    raise EnzymeError(f"unknown enzyme {name!r}")


def site_regex(site: str) -> re.Pattern[str]:
    """Compile an IUPAC site into an overlap-aware regex."""
    body = "".join(
        f"[{IUPAC[b]}]" if len(IUPAC[b]) > 1 else IUPAC[b] for b in site.upper()
    )
    return re.compile(f"(?=({body}))")


def find_sites(seq: str, site: str, *, circular: bool = False) -> list[int]:
    """0-based positions of `site` in `seq`, overlaps included.

    On a circular molecule the junction is searched too, so a site that
    straddles the origin is not missed.
    """
    seq = seq.upper()
    pattern = site_regex(site)
    n = len(seq)
    if circular and n >= len(site):
        haystack = seq + seq[: len(site) - 1]
    else:
        haystack = seq
    return sorted({m.start() % n for m in pattern.finditer(haystack)})


def cut_count(seq: str, enzyme: Enzyme, *, circular: bool = False) -> int:
    return len(find_sites(seq, enzyme.site, circular=circular))


def unique_cutters(
    seq: str, *, circular: bool = False, pool: dict[str, Enzyme] | None = None
) -> dict[str, int]:
    """Enzymes from `pool` that cut `seq` exactly once -> that position."""
    pool = pool or ENZYMES
    found: dict[str, int] = {}
    for name, enz in pool.items():
        hits = find_sites(seq, enz.site, circular=circular)
        if len(hits) == 1:
            found[name] = hits[0]
    return found


def non_cutters(
    seq: str, *, circular: bool = False, pool: dict[str, Enzyme] | None = None
) -> set[str]:
    """Enzymes from `pool` with no site in `seq` at all."""
    pool = pool or ENZYMES
    return {
        name
        for name, enz in pool.items()
        if not find_sites(seq, enz.site, circular=circular)
    }


def dam_blocked(context: str, site_offset: int, enzyme: Enzyme) -> bool:
    """Would Dam methylation block this site in this context?

    Dam methylates the A in GATC.  An enzyme flagged Dam-sensitive is only
    actually blocked when its recognition site overlaps a GATC, which for
    ClaI and XbaI depends on the neighbouring bases and for BclI is always
    true.  `context` must include at least 3 nt either side of the site.
    """
    if not enzyme.dam_sensitive:
        return False
    start = max(site_offset - 3, 0)
    stop = min(site_offset + enzyme.length + 3, len(context))
    region = context[start:stop].upper()
    for hit in find_sites(region, DAM_SITE):
        overlap_start = start + hit
        if (
            overlap_start < site_offset + enzyme.length
            and overlap_start + len(DAM_SITE) > site_offset
        ):
            return True
    return False


def dcm_blocked(context: str, site_offset: int, enzyme: Enzyme) -> bool:
    """Same idea for Dcm, which methylates the second C of CCWGG."""
    if not enzyme.dcm_sensitive:
        return False
    start = max(site_offset - 4, 0)
    stop = min(site_offset + enzyme.length + 4, len(context))
    region = context[start:stop].upper()
    for hit in find_sites(region, DCM_SITE):
        overlap_start = start + hit
        if (
            overlap_start < site_offset + enzyme.length
            and overlap_start + len(DCM_SITE) > site_offset
        ):
            return True
    return False


def compatible_ends(a: Enzyme, b: Enzyme) -> bool:
    """True if a fragment cut by `a` can ligate to one cut by `b`.

    Two blunt cutters are always compatible, which for directional cloning
    is the thing to avoid, not the thing to want.
    """
    if a.is_blunt and b.is_blunt:
        return True
    if a.overhang != b.overhang:
        return False
    if a.is_degenerate or b.is_degenerate:
        return False
    if a.overhang > 0:
        return a.overhang_seq == b.overhang_seq
    return revcomp(a.overhang_seq) == revcomp(b.overhang_seq)
