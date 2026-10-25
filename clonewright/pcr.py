"""In-silico PCR: where primers bind, what they amplify, and how to cycle.

The model is deliberately conservative about what counts as priming.  A
polymerase extends from a base-paired 3' terminus, so a primer with a
stable 3' anchor primes even when its 5' end dangles free — which is
exactly how a cloning primer with a restriction tail works — while a
primer that matches well in the middle but is mismatched at its last few
bases does not prime at all.  Binding is therefore judged on the
contiguous perfectly matched run ending at the primer's 3' base, not on
overall similarity.

This is a screen for mispriming, not a quantitative binding model.  It
will not predict yield, and it does not model mismatched base pairs
thermodynamically; a site is reported when its 3' anchor is long enough
and stable enough to prime at the annealing temperature in use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import thermo
from .seqio import Record, revcomp


@dataclass(frozen=True)
class Polymerase:
    """Vendor guidance for one polymerase.

    `ta_offset` is added to the lower primer melting temperature to get an
    annealing temperature.  Note that each vendor's rule is calibrated
    against that vendor's own melting-temperature calculator, which is not
    the SantaLucia model used elsewhere here; treat the suggestion as a
    starting point for a gradient, not a number to trust to the degree.
    """

    name: str
    seconds_per_kb: tuple[int, int]
    ta_offset: float
    proofreading: bool
    leaves_a_overhang: bool
    max_reliable_kb: float
    note: str = ""

    @property
    def end_type(self) -> str:
        return "3' A overhang" if self.leaves_a_overhang else "blunt"


POLYMERASES: dict[str, Polymerase] = {
    "taq": Polymerase(
        "Taq", (60, 60), -5.0, False, True, 5.0,
        "non-proofreading; products carry a 3' A, so they are for T/A "
        "cloning, not blunt ligation",
    ),
    "q5": Polymerase(
        "Q5", (20, 30), 1.0, True, False, 20.0,
        "high fidelity, blunt ends; stalls on uracil-containing template",
    ),
    "phusion": Polymerase(
        "Phusion", (15, 30), 3.0, True, False, 20.0,
        "high fidelity, blunt ends; the +3 rule applies to primers over "
        "20 nt, otherwise use the lower Tm directly",
    ),
    "pfu": Polymerase(
        "Pfu", (120, 120), -5.0, True, False, 5.0,
        "slow but accurate, blunt ends",
    ),
    "onetaq": Polymerase(
        "OneTaq", (60, 60), -5.0, False, True, 6.0,
        "Taq/proofreading blend; still leaves a 3' A",
    ),
}

DEFAULT_POLYMERASE = POLYMERASES["q5"]


def get_polymerase(name: str) -> Polymerase:
    try:
        return POLYMERASES[name.strip().lower()]
    except KeyError:
        raise KeyError(
            f"unknown polymerase {name!r}; known: "
            + ", ".join(sorted(POLYMERASES))
        ) from None


@dataclass(frozen=True)
class BindingSite:
    """One place a primer can prime, in template coordinates."""

    primer: str
    sequence: str
    template: str
    strand: int  # +1 extends towards higher coordinates, -1 towards lower
    start: int  # leftmost base of the matched footprint, 0-based
    end: int  # rightmost base of the footprint, inclusive
    anchor: int  # contiguous perfect match ending at the primer's 3' base
    mismatches: int  # within the footprint
    anchor_tm: float

    @property
    def footprint(self) -> int:
        return self.end - self.start + 1

    @property
    def three_prime(self) -> int:
        """Template coordinate of the primer's 3'-terminal base."""
        return self.end if self.strand > 0 else self.start

    @property
    def is_full_length(self) -> bool:
        return self.anchor >= self.footprint

    def describe(self) -> str:
        arrow = "-->" if self.strand > 0 else "<--"
        return (
            f"{self.primer} {arrow} {self.template}:{self.start + 1}-{self.end + 1}"
            f"  anchor {self.anchor} nt (Tm {self.anchor_tm:.1f}), "
            f"{self.mismatches} mismatch(es)"
        )


@dataclass(frozen=True)
class Product:
    template: str
    forward: BindingSite
    reverse: BindingSite
    length: int
    sequence: str

    @property
    def label(self) -> str:
        return f"{self.forward.primer} x {self.reverse.primer}"

    @property
    def is_clean(self) -> bool:
        return self.forward.mismatches == 0 and self.reverse.mismatches == 0


@dataclass
class Simulation:
    polymerase: Polymerase
    annealing_temp: float
    products: list[Product] = field(default_factory=list)
    sites: list[BindingSite] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def intended(self) -> Product | None:
        return self.products[0] if self.products else None

    @property
    def off_targets(self) -> list[Product]:
        return self.products[1:]

    def extension_seconds(self, length: int | None = None) -> tuple[int, int]:
        bp = length if length is not None else (
            self.intended.length if self.intended else 1000
        )
        lo, hi = self.polymerase.seconds_per_kb
        kb = bp / 1000.0
        return max(int(lo * kb), 10), max(int(hi * kb), 15)


# ---------------------------------------------------------------------------
# binding sites
# ---------------------------------------------------------------------------


def _anchor_length(primer: str, window: str) -> tuple[int, int]:
    """Contiguous perfect match ending at the 3' base, and total mismatches.

    `primer` and `window` are aligned at their right-hand ends, which is
    where the primer's 3' terminus sits.
    """
    anchor = 0
    counting = True
    mismatches = 0
    for p, t in zip(reversed(primer), reversed(window)):
        if p == t:
            if counting:
                anchor += 1
        else:
            counting = False
            mismatches += 1
    return anchor, mismatches


def find_binding_sites(
    primer: str,
    template: Record,
    *,
    name: str = "primer",
    min_anchor: int = 7,
    min_anchor_tm: float = 0.0,
    conditions: thermo.Conditions = thermo.DEFAULT_CONDITIONS,
) -> list[BindingSite]:
    """Every position where `primer` has a 3' anchor long enough to prime.

    Candidates are seeded on an exact match to the primer's 3'-terminal
    `min_anchor` bases, since priming requires those to be paired; the
    alignment is then extended 5' and scored.
    """
    primer = primer.upper()
    seq = template.seq
    n = len(seq)
    if len(primer) < min_anchor or n < min_anchor:
        return []

    sites: list[BindingSite] = []
    haystack = seq + seq[: len(primer)] if template.circular else seq

    for strand in (1, -1):
        probe = primer if strand > 0 else revcomp(primer)
        # The primer's 3' base lies at the right end of the probe when it
        # reads along the top strand, and at the left end when it is the
        # reverse complement.
        seed = probe[-min_anchor:] if strand > 0 else probe[:min_anchor]
        start = 0
        while True:
            hit = haystack.find(seed, start)
            if hit < 0:
                break
            start = hit + 1
            if strand > 0:
                three_prime = hit + min_anchor - 1
                overlap = min(len(probe), three_prime + 1)
                left = three_prime - overlap + 1
                window = haystack[left : three_prime + 1]
                anchor, mismatches = _anchor_length(probe[-overlap:], window)
                site_start, site_end = left, three_prime
            else:
                three_prime = hit
                overlap = min(len(probe), len(haystack) - three_prime)
                window = haystack[three_prime : three_prime + overlap]
                # Aligned at the primer's 3' end, which is the left end here.
                anchor, mismatches = _anchor_length(
                    probe[:overlap][::-1], window[::-1]
                )
                site_start, site_end = three_prime, three_prime + overlap - 1
            if anchor < min_anchor:
                continue
            if site_end >= n and not template.circular:
                continue
            matched = (
                probe[-anchor:] if strand > 0 else revcomp(probe[:anchor])
            )
            try:
                anchor_tm = thermo.tm(matched, conditions)
            except thermo.ThermoError:
                continue
            if anchor_tm < min_anchor_tm:
                continue
            sites.append(
                BindingSite(
                    primer=name,
                    sequence=primer,
                    template=template.name,
                    strand=strand,
                    start=site_start % n,
                    end=site_end % n if site_end >= n else site_end,
                    anchor=anchor,
                    mismatches=mismatches,
                    anchor_tm=anchor_tm,
                )
            )

    # Collapse duplicates arising from the circular overhang.
    unique: dict[tuple, BindingSite] = {}
    for s in sites:
        unique.setdefault((s.primer, s.strand, s.start, s.end), s)
    return sorted(unique.values(), key=lambda s: (s.start, s.strand))


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------


def _product_between(
    fwd: BindingSite, rev: BindingSite, template: Record, max_length: int
) -> Product | None:
    if fwd.strand <= 0 or rev.strand >= 0:
        return None
    n = len(template.seq)
    interior_start = fwd.end + 1
    interior_end = rev.start  # exclusive
    span = interior_end - interior_start
    if span < 0:
        if not template.circular:
            return None
        span += n
        if span < 0:
            return None
    length = len(fwd.sequence) + span + len(rev.sequence)
    if length > max_length:
        return None
    interior = template.window(interior_start, interior_end % n if n else 0)
    seq = fwd.sequence + interior + revcomp(rev.sequence)
    return Product(template.name, fwd, rev, length, seq)


def simulate(
    primers: dict[str, str],
    templates: list[Record],
    *,
    polymerase: Polymerase = DEFAULT_POLYMERASE,
    annealing_temp: float | None = None,
    conditions: thermo.Conditions = thermo.DEFAULT_CONDITIONS,
    max_length: int | None = None,
    anchor_margin: float = 5.0,
    min_anchor: int = 7,
) -> Simulation:
    """Run every primer against every template and enumerate the products.

    A site only counts if its 3' anchor would still be paired at the
    annealing temperature, so the annealing temperature has to be settled
    before the search: a primer that primes at 50 C may well not at 68 C.
    """
    notes: list[str] = []

    # Work out the annealing temperature from the part of each primer that
    # actually pairs with the template, which is the anchor at its best
    # site.  Taking it from the full-length primer instead would read the
    # restriction tail as though it annealed in the first cycle and put the
    # temperature 10-15 C too high, which is the classic way to design a
    # cloning PCR that never amplifies.
    if annealing_temp is None:
        effective: list[float] = []
        for name, seq in primers.items():
            best = 0.0
            for template in templates:
                for site in find_binding_sites(
                    seq, template, name=name, min_anchor=min_anchor,
                    conditions=conditions,
                ):
                    best = max(best, site.anchor_tm)
            if best:
                effective.append(best)
            else:
                try:
                    effective.append(thermo.tm(seq, conditions))
                except thermo.ThermoError:
                    pass
                notes.append(
                    f"{name} binds nowhere on the templates given; its "
                    "full-length Tm was used for the suggested Ta"
                )
        base = min(effective) if effective else 60.0
        annealing_temp = base + polymerase.ta_offset

    # The enumeration cap and the polymerase's working range are different
    # questions.  Capping enumeration at the polymerase limit would make an
    # over-long product disappear instead of being reported as too long,
    # leaving the user staring at "no product" for a product that exists.
    cap = max_length or 50_000
    sim = Simulation(polymerase=polymerase, annealing_temp=annealing_temp)
    sim.notes.extend(notes)

    for template in templates:
        sites: list[BindingSite] = []
        for name, seq in primers.items():
            sites += find_binding_sites(
                seq,
                template,
                name=name,
                min_anchor=min_anchor,
                min_anchor_tm=annealing_temp - anchor_margin,
                conditions=conditions,
            )
        sim.sites += sites
        for fwd in sites:
            for rev in sites:
                product = _product_between(fwd, rev, template, cap)
                if product is not None:
                    sim.products.append(product)

    # The intended product is the clean, full-length one; everything else
    # is ranked after it by how plausible it is.
    sim.products.sort(
        key=lambda p: (
            not (p.is_clean and p.forward.is_full_length and p.reverse.is_full_length),
            p.forward.mismatches + p.reverse.mismatches,
            -min(p.forward.anchor_tm, p.reverse.anchor_tm),
            p.length,
        )
    )

    if not sim.products:
        sim.notes.append(
            "no product: the primers do not both prime on any template at "
            f"{annealing_temp:.0f} C"
        )
    if sim.intended and sim.intended.length > polymerase.max_reliable_kb * 1000:
        sim.notes.append(
            f"{sim.intended.length} bp is beyond {polymerase.name}'s reliable "
            f"range of about {polymerase.max_reliable_kb:.0f} kb"
        )
    if polymerase.leaves_a_overhang:
        sim.notes.append(
            f"{polymerase.name} leaves a 3' A overhang, so this product is "
            "for T/A cloning; blunt ligation needs a proofreading enzyme or "
            "a polishing step"
        )
    return sim
