"""The E. coli expression vectors this package ships, and choosing between them.

Two separate things live here, and the split matters.

**The catalog** is what is known about each vector *as a product*: which
promoter drives it, what induces it, what the strain has to supply, which
antibiotic selects it, and what the expressed protein ends up carrying at
each terminus.  None of that is derivable from the bases, and none of it is
derived -- it is written down, with the vector, and it is the part a person
would otherwise look up in a manual.

**The choice** is not written down anywhere, because it depends on the
insert.  `shortlist` answers it by designing into every vector in the
catalog and reporting what actually happened: a vector that offers no
enzyme pair for this gene is not a candidate for this gene, however well it
suits the protein.  Nothing is scored on reputation.

The maps themselves are GenBank files under `data/vectors/`, with
annotations derived from sequence by `cassette`.  `tools/build_vector_maps.py`
builds them and says how.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import design as design_mod
from .seqio import Record, SequenceError, load

DATA = Path(__file__).resolve().parent / "data" / "vectors"


@dataclass(frozen=True)
class VectorSpec:
    """What is known about a vector that its sequence does not say."""

    name: str
    filename: str
    promoter: str
    induction: str
    # What the strain has to bring.  Empty when E. coli's own RNA
    # polymerase reads the promoter, which is the difference between a
    # vector that works in any strain and one that gives nothing outside
    # a DE3 lysogen.
    requires: str
    antibiotic: str
    n_terminal: str
    c_terminal: str
    purification: str
    note: str
    # The tag layout in a few words, for a menu row or a table cell. The
    # full sentences above are for a report, where there is room for them.
    summary: str = ""

    @property
    def path(self) -> Path:
        return DATA / self.filename

    def describe(self) -> str:
        return f"{self.name} ({self.promoter}, {self.antibiotic})"


CATALOG: dict[str, VectorSpec] = {
    "pET-28a": VectorSpec(
        name="pET-28a(+)",
        filename="pET-28a.gb",
        promoter="T7/lac",
        induction="0.1-1 mM IPTG at OD600 0.5-0.8",
        requires="T7 RNA polymerase from a lambda-DE3 lysogen",
        antibiotic="kanamycin",
        n_terminal="6xHis followed by a thrombin site",
        c_terminal="6xHis, downstream of the insert's own stop codon",
        purification="Ni-NTA on the N-terminal 6xHis; thrombin removes the tag",
        summary="N-terminal 6xHis, cleavable",
        note="the default first choice for a His-tagged protein. Kanamycin "
             "rather than ampicillin, which matters for long inductions: "
             "beta-lactamase is secreted and ampicillin plates go permissive",
    ),
    "pET-21b": VectorSpec(
        name="pET-21b(+)",
        filename="pET-21b.gb",
        promoter="T7/lac",
        induction="0.1-1 mM IPTG at OD600 0.5-0.8",
        requires="T7 RNA polymerase from a lambda-DE3 lysogen",
        antibiotic="ampicillin",
        n_terminal="nothing; the insert supplies its own start codon",
        c_terminal="6xHis, reached by removing the insert's stop codon",
        purification="Ni-NTA on the C-terminal 6xHis, which requires the "
                     "full-length product: a truncation loses the tag and "
                     "does not purify",
        summary="untagged, unless you take the C-terminal 6xHis",
        note="for a protein whose N terminus has to stay free -- a mature "
             "sequence, or anything where an N-terminal tag blocks folding "
             "or activity",
    ),
    "pGEX-4T-1": VectorSpec(
        name="pGEX-4T-1",
        filename="pGEX-4T-1.gb",
        promoter="tac",
        induction="0.1-1 mM IPTG at OD600 0.6",
        requires="",
        antibiotic="ampicillin",
        n_terminal="glutathione S-transferase (about 26 kDa) and a thrombin site",
        c_terminal="nothing",
        purification="glutathione sepharose; thrombin releases the protein "
                     "from GST",
        summary="N-terminal GST, 26 kDa, cleavable",
        note="the large soluble partner often rescues a protein that will not "
             "fold alone, at the cost of carrying 26 kDa through "
             "purification. No DE3 lysogen needed: tac is read by the host's "
             "own RNA polymerase",
    ),
    "pMAL-c5X": VectorSpec(
        name="pMAL-c5X",
        filename="pMAL-c5X.gb",
        promoter="tac",
        induction="0.3 mM IPTG at OD600 0.5",
        requires="",
        antibiotic="ampicillin",
        n_terminal="maltose binding protein (about 42 kDa) and a factor Xa site",
        c_terminal="nothing",
        purification="amylose resin; factor Xa releases the protein from MBP",
        summary="N-terminal MBP, 42 kDa, cleavable",
        note="the strongest solubility partner of the three, and the "
             "bulkiest. The c in c5X is cytoplasmic: there is no signal "
             "sequence, so the fusion stays in the cytoplasm",
    ),
}


class VectorError(ValueError):
    """Raised when a vector cannot be found or read."""


def available() -> list[str]:
    """Catalog names whose map file is actually present."""
    return [name for name, spec in CATALOG.items() if spec.path.exists()]


def _normalise(name: str) -> str:
    return (
        name.strip().lower()
        .replace("(+)", "").replace("+", "")
        .replace("-", "").replace("_", "").replace(" ", "")
    )


def spec_for(name: str) -> VectorSpec | None:
    """The catalog entry a name refers to, or None if it names no entry."""
    wanted = _normalise(name)
    for key, spec in CATALOG.items():
        if wanted in (_normalise(key), _normalise(spec.name)):
            return spec
    # A path to a bundled map, rather than its name.
    stem = _normalise(Path(name).stem)
    for key, spec in CATALOG.items():
        if stem == _normalise(key):
            return spec
    return None


def resolve(name: str) -> tuple[Record, VectorSpec | None]:
    """A vector by catalog name or by path, with its spec when it has one.

    A file path always wins over a catalog name.  Somebody who passes
    `./pET-28a.gb` means the file in front of them -- quite possibly their
    own corrected map -- and silently preferring the bundled copy of a
    vector with the same name would design against a different molecule
    than the one they are holding.
    """
    path = Path(name)
    if path.exists():
        try:
            record = load(path)
        except SequenceError as exc:
            raise VectorError(str(exc)) from exc
        return record, spec_for(path.stem)

    spec = spec_for(name)
    if spec is None:
        known = ", ".join(CATALOG)
        raise VectorError(
            f"{name!r} is neither a file nor a vector this package ships. "
            f"Bundled: {known}. Any GenBank, SnapGene .dna or FASTA map can "
            f"be given as a path instead"
        )
    if not spec.path.exists():
        raise VectorError(
            f"{spec.name} is in the catalog but its map is missing from "
            f"{DATA}. Build it with tools/build_vector_maps.py"
        )
    return load(spec.path), spec


# ---------------------------------------------------------------------------
# choosing a vector for one insert
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One row of the comparison: a vector, judged against one insert.

    A summary, deliberately -- not the `Design` it came from.  Carrying the
    design would carry its issues, and an issue that reaches an answer's
    provenance record has to appear in the answer under its own severity.
    A comparison across four vectors would then have to print every note
    about all four, which is not a comparison any more.

    So the detail stays where it belongs: this says what happened, and
    designing into the vector you picked says why.  Counts, not texts.
    """

    spec: VectorSpec
    ok: bool
    pair: str = ""
    mode: str = ""
    excision: int = 0
    warnings: int = 0
    notes: int = 0
    product_residues: int | None = None
    product_kda: float | None = None
    # Whether the expressed product actually carries vector-contributed
    # residues at either end -- which is what decides whether there is an
    # affinity tag on it to purify by.  A C-terminal tag the insert's own
    # stop codon sits in front of is annotated on the vector and absent
    # from the protein, and this is the field that tells the two apart.
    tagged: bool = False
    refusal: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"{self.spec.name}: refused -- {self.refusal}"
        return (
            f"{self.spec.name}: {self.pair} as {self.mode}, "
            f"{self.warnings} warning(s)"
        )


def shortlist(
    insert: Record,
    *,
    names: list[str] | None = None,
    opts: design_mod.Options | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Design `insert` into every bundled vector; return (usable, rejected).

    The ranking is by what the design produced, in this order: it worked,
    it produced no warnings, and it needs the least vector sequence
    removed.  Nothing about the vector's popularity enters into it, and a
    vector that cannot take this particular gene -- because its only
    useful enzymes cut inside it -- is rejected regardless of how well it
    would otherwise suit the protein.

    `opts.mode` is left at whatever the caller set, which should normally
    be "auto": each vector then gets the cloning mode it is built for,
    and forcing one mode across a mixed catalog would reject vectors for
    not being the kind of vector they never claimed to be.
    """
    from . import expression as expression_mod

    usable: list[Candidate] = []
    rejected: list[Candidate] = []
    for name in names or available():
        spec = spec_for(name)
        if spec is None:
            continue
        try:
            vector, _ = resolve(name)
        except VectorError as exc:
            rejected.append(Candidate(spec=spec, ok=False, refusal=str(exc)))
            continue
        try:
            d = design_mod.design(insert, vector, opts)
        except design_mod.DesignFailed as exc:
            rejected.append(Candidate(spec=spec, ok=False, refusal=str(exc)))
            continue
        if not d.ok:
            rejected.append(Candidate(
                spec=spec, ok=False,
                refusal="; ".join(i.text for i in d.errors),
            ))
            continue
        made = expression_mod.product_of(d)
        usable.append(Candidate(
            spec=spec,
            ok=True,
            pair=d.pair.label,
            mode=d.options.mode,
            excision=d.pair.excision,
            warnings=sum(1 for i in d.issues if i.level == "warning"),
            notes=sum(1 for i in d.issues if i.level == "note"),
            product_residues=len(made) if made is not None else None,
            product_kda=made.mass_kda if made is not None else None,
            tagged=bool(made is not None and (made.leader or made.trailer)),
        ))

    # Fewest warnings first; then a product that can be purified by an
    # affinity tag ahead of one that cannot; then the least vector
    # sequence removed.
    #
    # The middle term is about the protein, not about the plasmid's
    # reputation.  pET-21b and pET-28a tie on warnings for most inserts,
    # and pET-21b removes slightly less vector -- so on size alone it won,
    # and what it wins with is an untagged product: its 6xHis is C-terminal
    # and sits past the insert's own stop codon, so nothing is translated
    # onto the protein and there is nothing to run over a nickel column.
    # Ranking a construct you cannot purify above one you can is not
    # frugality.
    usable.sort(key=lambda c: (c.warnings, not c.tagged, c.excision))
    return usable, rejected
