"""E. coli strains, and which one each half of the job belongs in.

Cloning and expressing are two different jobs wanted from two different
bacteria, and a construct normally passes through both: it is built and
sequence-verified in a cloning strain, stored there as a glycerol stock,
and only then transformed fresh into an expression strain each time
protein is wanted.

The split is not ceremony.

* A cloning strain is *recA* and *endA* deficient.  recA deficiency stops
  the plasmid recombining with itself, which matters for anything
  carrying repeats; endA deficiency removes endonuclease I, which is why
  a DH5-alpha miniprep is clean and a BL21 miniprep smears.
* An expression strain is protease deficient -- *lon* and *ompT* -- so
  the protein that is made survives long enough to be purified.
* Those are different mutations in different strains and nobody has to
  choose between them, because the plasmid can simply be moved.

The check that actually catches people is narrower and harsher.  **A T7
promoter does nothing without T7 RNA polymerase**, and no E. coli makes
that enzyme: it comes from the lambda-DE3 prophage.  So a pET-style
vector in plain BL21 gives no protein at all -- not a poor yield, none --
and the plate looks perfectly normal, because the antibiotic resistance
works fine. "BL21" and "BL21(DE3)" differ by one lysogen and by whether
the experiment works.

The second trap is quieter: several expression strains carry a *second*
plasmid of their own -- pLysS and pRARE are both chloramphenicol
resistant -- so the plate needs two antibiotics, and selecting for only
the vector's lets the strain throw away the thing you chose it for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .design import Issue, detect_host, driving_promoter
from .seqio import Record

# Promoters that need a polymerase the host has to supply, and what
# supplies it.  Everything else on this list is read by the host's own
# RNA polymerase and works in any strain.
ORTHOGONAL_PROMOTERS: dict[str, str] = {
    "t7": "T7 RNA polymerase, from a lambda-DE3 lysogen",
    "sp6": "SP6 RNA polymerase, which no standard E. coli carries",
}

# Promoters E. coli reads with its own RNA polymerase.  Worth naming
# rather than lumping with "no promoter found", because the two look the
# same to a search and mean opposite things: one is a vector that will
# express in any strain, the other is a map that has not been annotated.
# pCold is the case that matters here -- its cspA promoter needs no DE3
# lysogen, so plain BL21 works where a pET vector would give nothing.
HOST_PROMOTERS: dict[str, str] = {
    "cspa": "the cold-shock cspA promoter, read by the host's own RNA "
            "polymerase and induced by dropping the culture to 15 C",
    "tac": "a hybrid trp/lac promoter, read by the host's own RNA polymerase",
    "trc": "a hybrid trp/lac promoter, read by the host's own RNA polymerase",
    "arabad": "the arabinose promoter, read by the host's own RNA polymerase",
    "pbad": "the arabinose promoter, read by the host's own RNA polymerase",
    "rhap": "the rhamnose promoter, read by the host's own RNA polymerase",
}


@dataclass(frozen=True)
class Strain:
    """One E. coli strain and the properties that decide what it is for."""

    name: str
    role: str  # "cloning", "expression", or "both"
    t7_polymerase: bool = False
    reca_deficient: bool = False
    enda_deficient: bool = False
    protease_deficient: bool = False
    rare_codon_trnas: bool = False
    oxidising_cytoplasm: bool = False
    tight_repression: bool = False
    # A resident plasmid the strain carries, which has to be selected for
    # alongside the vector or the strain loses it.
    resident_marker: str = ""
    note: str = ""

    @property
    def good_for_minipreps(self) -> bool:
        return self.enda_deficient

    @property
    def stores_plasmids_well(self) -> bool:
        return self.reca_deficient and self.enda_deficient

    def describe(self) -> str:
        bits = []
        if self.t7_polymerase:
            bits.append("T7 RNAP (DE3)")
        if self.reca_deficient:
            bits.append("recA-")
        if self.enda_deficient:
            bits.append("endA-")
        if self.protease_deficient:
            bits.append("lon-/ompT-")
        if self.rare_codon_trnas:
            bits.append("rare tRNAs")
        if self.oxidising_cytoplasm:
            bits.append("oxidising cytoplasm")
        return f"{self.name} ({', '.join(bits)})" if bits else self.name


STRAINS: dict[str, Strain] = {
    "dh5alpha": Strain(
        "DH5-alpha", "cloning",
        reca_deficient=True, enda_deficient=True,
        note="the default cloning and storage host: recA1 endA1, so the "
             "plasmid stays intact and minipreps are clean",
    ),
    "top10": Strain(
        "TOP10", "cloning",
        reca_deficient=True, enda_deficient=True,
        note="equivalent to DH5-alpha for most purposes; tolerates "
             "unstable inserts a little better",
    ),
    "stbl3": Strain(
        "Stbl3", "cloning",
        reca_deficient=True, enda_deficient=True,
        note="for constructs with direct repeats, such as lentiviral LTRs, "
             "which recombine out of ordinary strains",
    ),
    "bl21": Strain(
        "BL21", "expression",
        protease_deficient=True,
        note="no DE3 lysogen, so no T7 RNA polymerase. Only useful for a "
             "vector whose promoter E. coli reads by itself",
    ),
    "bl21de3": Strain(
        "BL21(DE3)", "expression",
        t7_polymerase=True, protease_deficient=True,
        note="the standard T7 expression host: lon- and ompT- for protein "
             "survival, lambda-DE3 for T7 RNA polymerase under lacUV5",
    ),
    "bl21de3plyss": Strain(
        "BL21(DE3)pLysS", "expression",
        t7_polymerase=True, protease_deficient=True, tight_repression=True,
        resident_marker="chloramphenicol",
        note="T7 lysozyme suppresses basal T7 transcription before "
             "induction, which is what makes a toxic product clonable",
    ),
    "rosettade3": Strain(
        "Rosetta(DE3)", "expression",
        t7_polymerase=True, protease_deficient=True, rare_codon_trnas=True,
        resident_marker="chloramphenicol",
        note="carries pRARE, supplying tRNAs for codons E. coli uses "
             "rarely; for genes from organisms with a different codon bias",
    ),
    "shuffle": Strain(
        "SHuffle T7", "expression",
        t7_polymerase=True, protease_deficient=True,
        oxidising_cytoplasm=True,
        note="oxidising cytoplasm plus DsbC, so disulfide bonds can form; "
             "for secreted and extracellular domains, which otherwise "
             "misfold in the reducing cytoplasm",
    ),
    "origami2de3": Strain(
        "Origami 2(DE3)", "expression",
        t7_polymerase=True, protease_deficient=True,
        oxidising_cytoplasm=True,
        note="trxB/gor mutant; the other common route to disulfide bonds",
    ),
}

DEFAULT_CLONING = "dh5alpha"
DEFAULT_EXPRESSION = "bl21de3"


def get(name: str) -> Strain:
    key = (
        name.strip().lower()
        .replace("-", "").replace("_", "").replace(" ", "")
        .replace("(", "").replace(")", "").replace("α", "alpha")
    )
    if key in STRAINS:
        return STRAINS[key]
    raise KeyError(
        f"unknown strain {name!r}; known: " + ", ".join(sorted(STRAINS))
    )


# ---------------------------------------------------------------------------
# what the vector needs from the host
# ---------------------------------------------------------------------------


def host_promoter(vector: Record) -> tuple[str, str] | None:
    """A promoter the host reads itself, needing no supplied polymerase."""
    for f in vector.features:
        name = f.name.lower()
        if "promoter" not in name and f.kind.lower() != "promoter":
            continue
        if "laci" in name or "ampr" in name:
            continue
        for key, description in HOST_PROMOTERS.items():
            if name.startswith(key) or f"{key} promoter" in name:
                return key.upper(), description
    return None


def promoter_requirement(vector: Record) -> tuple[str, str] | None:
    """Which orthogonal polymerase this vector's promoter needs, if any.

    Only the promoter that actually drives the insert can create this
    requirement, so the question is asked of `driving_promoter` rather
    than answered by a second, independent scan of the annotations.

    Scanning for itself was the bug: any promoter feature called T7
    anywhere on the map produced a demand for a lambda-DE3 lysogen, so
    pcDNA3.1 and pLJM1-EGFP -- both mammalian -- asked for one on the
    strength of a T7 site that exists for in-vitro transcription and
    sequencing. A T7 terminator was already excluded; a T7 promoter
    that drives nothing was not.
    """
    driver = driving_promoter(vector)
    if driver is None:
        return None
    name = driver.name.lower()
    for key, supplied_by in ORTHOGONAL_PROMOTERS.items():
        if name.startswith(key) or f"{key} promoter" in name:
            return key.upper(), supplied_by
    return None


@dataclass
class HostPlan:
    """Which strain does which half, and whether that works."""

    cloning: Strain
    expression: Strain | None
    vector: Record
    promoter: tuple[str, str] | None = None
    antibiotics: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def plan_hosts(
    vector: Record,
    *,
    cloning: Strain | None = None,
    expression: Strain | None = None,
    vector_antibiotic: str = "",
) -> HostPlan:
    """Check a construct against the strains it is going to live in."""
    cloning = cloning or get(DEFAULT_CLONING)
    plan = HostPlan(cloning=cloning, expression=expression, vector=vector)
    plan.promoter = promoter_requirement(vector)
    if vector_antibiotic:
        plan.antibiotics.append(vector_antibiotic)

    if cloning.role == "expression":
        plan.issues.append(
            Issue(
                "warning",
                f"{cloning.name} is an expression strain being used for "
                "cloning. It is endA+, so minipreps carry nuclease and "
                "sequencing is poorer, and any leakiness selects for "
                "plasmids that have lost the insert",
            )
        )
    elif not cloning.stores_plasmids_well:
        plan.issues.append(
            Issue(
                "warning",
                f"{cloning.name} is not both recA- and endA-, so it is a "
                "poor place to keep the verified construct",
            )
        )
    else:
        plan.issues.append(
            Issue(
                "note",
                f"{cloning.name}: {cloning.note}",
            )
        )

    if expression is None:
        return plan

    # The check that matters.
    if plan.promoter is not None:
        label, supplied_by = plan.promoter
        if label == "T7" and not expression.t7_polymerase:
            plan.issues.append(
                Issue(
                    "error",
                    f"the vector is driven by a {label} promoter, which "
                    f"needs {supplied_by}. {expression.name} does not carry "
                    "it, so there will be no protein at all -- and the "
                    "plate will look completely normal, because the "
                    "antibiotic resistance is unaffected. Use BL21(DE3) or "
                    "another DE3 lysogen",
                )
            )
        elif label == "SP6":
            plan.issues.append(
                Issue(
                    "error",
                    f"the vector is driven by an {label} promoter, which "
                    f"needs {supplied_by}. This is an in vitro transcription "
                    "promoter, not an E. coli expression one",
                )
            )
        else:
            plan.issues.append(
                Issue(
                    "note",
                    f"{label} promoter and {expression.name} carries "
                    f"{supplied_by}, so transcription of the insert is "
                    "induced rather than constitutive",
                )
            )
    else:
        own = host_promoter(vector)
        if own is not None:
            label, description = own
            plan.issues.append(
                Issue(
                    "note",
                    f"the insert is driven by {description}. Nothing has to "
                    "be supplied by the strain, so a DE3 lysogen is not "
                    f"required here: plain BL21 would express this as well "
                    f"as {expression.name} does"
                    + (
                        ". Induction is the temperature drop as much as the "
                        "IPTG: expressed at 37 C this vector gives very "
                        "little"
                        if label == "CSPA"
                        else ""
                    ),
                )
            )
        else:
            # A third answer, and the one the T7 bug was hiding.  While
            # any T7-named promoter anywhere on the map counted, a
            # mammalian vector took the orthogonal branch and was told
            # to use a DE3 lysogen.  With that fixed it would otherwise
            # fall through to "no promoter was found", which is equally
            # untrue: pcDNA3.1's CMV promoter is annotated, found, and
            # simply not something E. coli transcribes.
            driver = driving_promoter(vector)
            host = detect_host(vector)
            if driver is not None and host in ("mammalian", "yeast"):
                plan.issues.append(
                    Issue(
                        "warning",
                        f"the insert is driven by the {driver.name}, which "
                        f"is a {host} promoter: no E. coli strain "
                        f"transcribes it, so {expression.name} will carry "
                        "this plasmid and express nothing from the cassette. "
                        "The E. coli half of this job is cloning and "
                        "sequence verification only -- expression happens "
                        f"after the construct is put into {host} cells",
                    )
                )
            else:
                plan.issues.append(
                    Issue(
                        "warning",
                        "no promoter was found in the vector's annotations, "
                        "neither one the host reads itself nor one needing a "
                        "supplied polymerase. Either the map is incomplete or "
                        "expression has not been checked",
                    )
                )

    if expression.role == "cloning":
        plan.issues.append(
            Issue(
                "error",
                f"{expression.name} is a cloning strain: it is not protease "
                "deficient, and it carries no T7 polymerase",
            )
        )
    if expression.resident_marker:
        plan.antibiotics.append(expression.resident_marker)
        plan.issues.append(
            Issue(
                "warning",
                f"{expression.name} carries a resident plasmid selected with "
                f"{expression.resident_marker}, so plates and cultures need "
                "both that and the vector's own antibiotic. Selecting for "
                "only one lets the strain drop the other",
            )
        )
    if expression.note:
        plan.issues.append(Issue("note", f"{expression.name}: {expression.note}"))

    plan.issues.append(
        Issue(
            "note",
            f"transform {expression.name} fresh from the verified "
            f"{cloning.name} stock each time, rather than keeping a working "
            "stock of it. An expression strain is where the construct is "
            "used, not where it lives",
        )
    )
    return plan


def suggest_expression_strain(
    *,
    rare_codons: bool = False,
    disulfides: bool = False,
    toxic: bool = False,
) -> Strain:
    """The expression strain a construct's own properties point to."""
    if disulfides:
        return get("shuffle")
    if rare_codons:
        return get("rosettade3")
    if toxic:
        return get("bl21de3plyss")
    return get(DEFAULT_EXPRESSION)
