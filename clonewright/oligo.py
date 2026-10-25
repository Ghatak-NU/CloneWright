"""Primer structure and the quality checks applied to one oligo.

A cloning primer is four pieces::

    5'-[spacer][restriction site][pad][annealing region]-3'

Only the annealing region base-pairs with the template in the first round
of PCR, so that is the part whose melting temperature sets the annealing
temperature.  Reporting the melting temperature of the whole oligo instead
is the single most common way a cloning primer set goes wrong, so both
numbers are kept and the design code is explicit about which it uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import thermo
from .seqio import gc_fraction, revcomp


# Hairpin loop initiation penalties at 37 C, kcal/mol (Turner/SantaLucia
# folding rules).  Closing a stem into a loop costs entropy, and a tight
# loop costs most; ignoring this makes every short-loop hairpin look far
# more stable than it is.  Values between tabulated sizes are interpolated
# and larger loops extrapolated logarithmically.
_LOOP_PENALTY = {
    3: 3.5, 4: 3.5, 5: 3.3, 6: 4.0, 7: 4.2, 8: 4.3, 9: 4.5, 10: 4.6,
    12: 5.0, 14: 5.1, 16: 5.3, 18: 5.5, 20: 5.7, 25: 6.1, 30: 6.3,
}


def loop_penalty(n: int) -> float:
    """Free-energy cost of closing a hairpin loop of `n` unpaired bases."""
    if n < 3:
        return 10.0  # sterically impossible; price it out of consideration
    if n in _LOOP_PENALTY:
        return _LOOP_PENALTY[n]
    sizes = sorted(_LOOP_PENALTY)
    if n > sizes[-1]:
        import math

        return _LOOP_PENALTY[sizes[-1]] + 1.75 * 0.0019872 * 310.15 * math.log(
            n / sizes[-1]
        )
    lo = max(s for s in sizes if s < n)
    hi = min(s for s in sizes if s > n)
    span = hi - lo
    return _LOOP_PENALTY[lo] + (_LOOP_PENALTY[hi] - _LOOP_PENALTY[lo]) * (n - lo) / span


@dataclass(frozen=True)
class Duplex:
    """A perfectly matched stretch between two oligos (or one with itself).

    For a hairpin, `dg` is the net cost of folding: the stem's duplex free
    energy plus the loop closure penalty.  For a dimer there is no loop and
    the two are the same.
    """

    length: int
    seq: str
    dg: float
    involves_3prime: bool
    loop: int = 0
    stem_dg: float = 0.0
    offset: int = -1  # where the pairing starts in the oligo

    @property
    def is_concerning(self) -> bool:
        """Whether this pairing is strong enough to actually cost yield.

        A 3'-end pairing matters more than an internal one because it can
        be extended, but four complementary bases worth about -2.5 kcal/mol
        come apart at any annealing temperature worth using, so the
        threshold sits where a structure starts competing with the template.
        """
        if self.involves_3prime and self.length >= 5 and self.dg <= -4.0:
            return True
        return self.length >= 8 or self.dg <= -9.0

    @property
    def is_folded(self) -> bool:
        """For hairpins: whether folding is favourable at all."""
        return self.dg < 0


THREE_PRIME_FACTOR = 3.0
"""How much worse a 3'-end pairing is than an internal one.

A duplex at the 3' end can be extended by the polymerase, converting the
primer into a dead product; an internal one only sequesters it briefly.
"""


def _maximal_runs(a: str, b: str, min_len: int) -> list[tuple[int, int, int]]:
    """Every maximal matching run between `a` and `b`.

    Yields (start_in_a, start_in_b, length).  A run is maximal when it
    cannot be extended by one more base on the right, which keeps the
    candidate set small without losing any distinct pairing.
    """
    if not a or not b:
        return []
    runs: list[tuple[int, int, int]] = []
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
        for j in range(1, len(b) + 1):
            length = cur[j]
            if length >= min_len and (
                i == len(a) or j == len(b) or a[i] != b[j]
            ):
                runs.append((i - length, j - length, length))
        prev = cur
    return runs


def worst_duplex(
    a: str, b: str, *, three_prime_window: int = 5, min_len: int = 4
) -> Duplex:
    """The most harmful perfect duplex formable between oligos `a` and `b`.

    If `a` contains a run X and `b` contains its reverse complement, then
    revcomp(b) contains X too, so shared runs between `a` and revcomp(b)
    are exactly the duplexes the two can form.

    Candidates are ranked by severity, free energy weighted by whether the
    pairing reaches a 3' end, rather than by raw length.  Ranking by
    length alone breaks ties arbitrarily between equally long duplexes,
    which once made a primer carrying two dimers look better than one
    carrying a single weaker dimer.
    """
    a, b = a.upper(), b.upper()
    rb = revcomp(b)
    best = Duplex(0, "", 0.0, False)
    best_severity = 0.0
    for start_a, start_rb, length in _maximal_runs(a, rb, min_len):
        seq = a[start_a : start_a + length]
        dg = thermo.delta_g(seq)
        if dg >= 0:
            continue
        # The pairing uses a's 3' end if it runs into a's last bases, or
        # b's 3' end, which sits at the start of revcomp(b).
        three = (
            start_a + length >= len(a) - three_prime_window + 1
            or start_rb < three_prime_window
        )
        severity = -dg * (THREE_PRIME_FACTOR if three else 1.0)
        if severity > best_severity:
            best_severity = severity
            best = Duplex(length, seq, dg, three, stem_dg=dg, offset=start_a)
    return best


def hairpin(seq: str, *, min_stem: int = 3, min_loop: int = 3) -> Duplex:
    """Most stable self-folding structure for one oligo.

    Candidates are ranked by net folding free energy, stem plus loop, not
    by stem length: a short G/C stem closed by a comfortable loop can be
    more stable than a longer stem strained into a triloop.
    """
    seq = seq.upper()
    best = Duplex(0, "", 0.0, False)
    best_severity = 0.0
    n = len(seq)
    for i in range(n):
        for j in range(i + 2 * min_stem + min_loop, n + 1):
            for stem in range(min_stem, (j - i - min_loop) // 2 + 1):
                left = seq[i : i + stem]
                right = seq[j - stem : j]
                if left != revcomp(right):
                    continue
                loop = j - i - 2 * stem
                stem_dg = thermo.delta_g(left)
                net = stem_dg + loop_penalty(loop)
                if net >= 0:
                    continue
                three = j >= n - 4
                severity = -net * (THREE_PRIME_FACTOR if three else 1.0)
                if severity > best_severity:
                    best_severity = severity
                    best = Duplex(stem, left, net, three, loop, stem_dg, i)
    return best


@dataclass
class Oligo:
    """One designed primer."""

    name: str
    spacer: str = ""
    site: str = ""
    pad: str = ""
    anneal: str = ""
    conditions: thermo.Conditions = thermo.DEFAULT_CONDITIONS

    @property
    def tail(self) -> str:
        return f"{self.spacer}{self.site}{self.pad}"

    @property
    def sequence(self) -> str:
        return f"{self.tail}{self.anneal}"

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def tm_anneal(self) -> float:
        """Melting temperature of the template-binding part. Sets Ta."""
        return thermo.tm(self.anneal, self.conditions)

    @property
    def tm_full(self) -> float:
        """Melting temperature of the whole oligo, relevant from cycle 3 on."""
        return thermo.tm(self.sequence, self.conditions)

    @property
    def gc(self) -> float:
        return gc_fraction(self.anneal)

    @property
    def gc_full(self) -> float:
        return gc_fraction(self.sequence)

    @property
    def three_prime_dg(self) -> float:
        return thermo.three_prime_dg(self.anneal)

    @property
    def gc_clamp(self) -> int:
        return thermo.gc_clamp(self.anneal)

    @property
    def longest_run(self) -> int:
        return thermo.longest_run(self.anneal)

    @property
    def self_dimer(self) -> Duplex:
        return worst_duplex(self.sequence, self.sequence)

    @property
    def hairpin(self) -> Duplex:
        return hairpin(self.sequence)

    def _within_site(self, d: Duplex) -> bool:
        """Does this pairing lie inside the restriction site?

        Recognition sites are palindromes, so a primer always pairs with
        itself across its own site.  Every directional cloning primer has
        this and no choice of annealing region changes it, so it is not a
        defect and must not drive the ranking.
        """
        if d.offset < 0 or not self.site:
            return False
        site_start = len(self.spacer)
        site_end = site_start + len(self.site)
        return d.offset >= site_start - 1 and d.offset + d.length <= site_end + 1

    @property
    def structure_dg(self) -> float:
        """Raw folding free energy of hairpin plus self-dimer, kcal/mol."""
        return min(self.hairpin.dg, 0.0) + min(self.self_dimer.dg, 0.0)

    @property
    def structure_penalty(self) -> float:
        """Folding burden weighted by whether it can actually hurt the PCR.

        Structure at the 3' end is extendable and genuinely costs yield;
        structure confined to the 5' tail only sequesters primer briefly
        and is mild; a pairing across the palindromic restriction site is
        unavoidable and is discounted hardest.
        """
        tail_len = len(self.tail)
        total = 0.0
        for d in (self.hairpin, self.self_dimer):
            if d.dg >= 0:
                continue
            if d.involves_3prime:
                weight = 3.0
            elif self._within_site(d):
                weight = 0.1
            elif d.offset >= 0 and d.offset + d.length <= tail_len:
                weight = 0.4
            else:
                weight = 1.0
            total += -d.dg * weight
        return total

    @property
    def dimer_is_site_palindrome(self) -> bool:
        return self._within_site(self.self_dimer)

    def formatted(self) -> str:
        """The oligo with its parts separated, for ordering and for reading."""
        parts = [p for p in (self.spacer, self.site, self.pad, self.anneal) if p]
        return "5'-" + " ".join(parts) + "-3'"
