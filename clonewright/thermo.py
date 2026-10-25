"""Nearest-neighbour duplex thermodynamics.

Parameters are the SantaLucia (1998) unified set, PNAS 95:1460-1465, which
are defined at 1 M Na+.  Salt is then corrected onto that reference in two
steps:

  * Mg2+ and dNTPs are folded into an equivalent monovalent concentration,
    [Na+]eq = [mono+] + 120*sqrt([Mg2+]free), with [Mg2+]free taken as
    [Mg2+] - [dNTP] because dNTPs chelate Mg2+ roughly 1:1.
    von Ahsen et al. (2001) Clin Chem 47:1956.
  * That equivalent is applied with the composition-dependent monovalent
    correction of Owczarzy et al. (2004) Biochemistry 43:3537.

An earlier draft applied the Owczarzy (2008) divalent equation directly.
It was removed: it returned melting temperatures above the 1 M Na+
reference for ordinary PCR conditions, which cannot be right, and the sign
convention on its coefficients could not be confirmed here.  The route
above is the one whose behaviour is verified by the test suite.

Everything here operates on a single strand paired with its perfect
complement.  Mismatched duplexes are out of scope; the design code only
ever asks about perfectly matched annealing regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R_GAS = 1.9872  # cal / (mol K)
ZERO_C = 273.15

# kcal/mol.  Keys are 5'->3' dinucleotides on the given strand; a step and
# its reverse complement share a value, so both spellings are listed.
NN_DH: dict[str, float] = {
    "AA": -7.9, "TT": -7.9,
    "AT": -7.2,
    "TA": -7.2,
    "CA": -8.5, "TG": -8.5,
    "GT": -8.4, "AC": -8.4,
    "CT": -7.8, "AG": -7.8,
    "GA": -8.2, "TC": -8.2,
    "CG": -10.6,
    "GC": -9.8,
    "GG": -8.0, "CC": -8.0,
}

# cal / (mol K)
NN_DS: dict[str, float] = {
    "AA": -22.2, "TT": -22.2,
    "AT": -20.4,
    "TA": -21.3,
    "CA": -22.7, "TG": -22.7,
    "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0,
    "GA": -22.2, "TC": -22.2,
    "CG": -27.2,
    "GC": -24.4,
    "GG": -19.9, "CC": -19.9,
}

INIT_GC = (0.1, -2.8)   # dH kcal/mol, dS cal/mol/K for a terminal G:C
INIT_AT = (2.3, 4.1)    # ... and for a terminal A:T


@dataclass(frozen=True)
class Conditions:
    """PCR conditions.  Concentrations in molar except where noted."""

    primer: float = 250e-9    # each primer, 0.25 uM is a typical PCR
    na: float = 50e-3
    k: float = 0.0
    tris: float = 10e-3       # Tris buffer contributes ~half its molarity
    mg: float = 1.5e-3
    dntp: float = 0.8e-3      # chelates Mg2+ roughly 1:1

    @property
    def monovalent(self) -> float:
        return self.na + self.k + self.tris / 2.0

    @property
    def mg_free(self) -> float:
        return max(self.mg - self.dntp, 0.0)

    @property
    def sodium_equivalent(self) -> float:
        """Monovalent concentration that would stabilise the duplex equally.

        von Ahsen (2001): 120*sqrt([Mg2+]free) in mM.  The square root is
        taken in mM and converted back to molar, so the constant keeps its
        published units.
        """
        mg_mm = self.mg_free * 1000.0
        return self.monovalent + (120.0 * math.sqrt(mg_mm)) / 1000.0


DEFAULT_CONDITIONS = Conditions()


class ThermoError(ValueError):
    pass


def _require_unambiguous(seq: str) -> str:
    seq = seq.upper()
    if not seq:
        raise ThermoError("empty sequence")
    bad = sorted(set(seq) - set("ACGT"))
    if bad:
        raise ThermoError(
            "nearest-neighbour parameters need unambiguous DNA; "
            f"got {', '.join(bad)}"
        )
    return seq


def enthalpy_entropy(seq: str) -> tuple[float, float]:
    """Return (dH kcal/mol, dS cal/mol/K) for seq paired with its complement."""
    seq = _require_unambiguous(seq)
    if len(seq) < 2:
        raise ThermoError("need at least 2 nt")
    dh = ds = 0.0
    for i in range(len(seq) - 1):
        step = seq[i : i + 2]
        dh += NN_DH[step]
        ds += NN_DS[step]
    for end in (seq[0], seq[-1]):
        init_dh, init_ds = INIT_GC if end in "GC" else INIT_AT
        dh += init_dh
        ds += init_ds
    return dh, ds


def delta_g(seq: str, temp_c: float = 37.0) -> float:
    """Duplex free energy in kcal/mol at 1 M Na+ (negative = stable)."""
    dh, ds = enthalpy_entropy(seq)
    return dh - (temp_c + ZERO_C) * ds / 1000.0


def is_self_complementary(seq: str) -> bool:
    from .seqio import revcomp

    return seq.upper() == revcomp(seq.upper())


def _tm_1m_na(seq: str, primer_conc: float) -> float:
    dh, ds = enthalpy_entropy(seq)
    if is_self_complementary(seq):
        effective = primer_conc
    else:
        # Non-self-complementary strands in excess over their template:
        # the usual CT/4 term.
        effective = primer_conc / 4.0
    if effective <= 0:
        raise ThermoError("primer concentration must be positive")
    return (dh * 1000.0) / (ds + R_GAS * math.log(effective)) - ZERO_C


def _correct_monovalent(tm_1m: float, fgc: float, mono: float) -> float:
    """Owczarzy 2004 eq. 22 (sequence-composition-dependent)."""
    ln_na = math.log(mono)
    inv = (
        1.0 / (tm_1m + ZERO_C)
        + (4.29 * fgc - 3.95) * 1e-5 * ln_na
        + 9.40e-6 * ln_na * ln_na
    )
    return 1.0 / inv - ZERO_C


def tm(seq: str, conditions: Conditions = DEFAULT_CONDITIONS) -> float:
    """Melting temperature in degrees C under `conditions`.

    Mg2+ is folded into an equivalent monovalent concentration first, then
    the composition-dependent monovalent correction is applied against the
    1 M Na+ reference.  The result is always at or below the 1 M Na+ value,
    which is the sanity property the test suite pins down.
    """
    seq = _require_unambiguous(seq)
    from .seqio import gc_fraction

    tm_1m = _tm_1m_na(seq, conditions.primer)
    salt = conditions.sodium_equivalent
    if salt <= 0:
        return tm_1m
    return _correct_monovalent(tm_1m, gc_fraction(seq), salt)


def three_prime_dg(seq: str, window: int = 5) -> float:
    """Free energy of the last `window` bases, at 1 M Na+ and 37 C.

    A very stable 3' end primes off partial matches, so this is capped in
    the design scoring rather than minimised.
    """
    seq = _require_unambiguous(seq)
    tail = seq[-window:] if len(seq) >= window else seq
    if len(tail) < 2:
        return 0.0
    return delta_g(tail)


def longest_run(seq: str) -> int:
    """Length of the longest single-base run."""
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def gc_clamp(seq: str) -> int:
    """Number of G/C in the last 5 bases."""
    return sum(1 for b in seq[-5:].upper() if b in "GC")
