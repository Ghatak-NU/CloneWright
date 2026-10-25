"""Rendering a design, and the two verification plans, as bench-ready text.

Terminal output: every renderer here returns a string that goes to a human
and never becomes another tool's input.  Nothing in this module computes a
number -- each one is read off an object a Layer 1 function returned --
which is what lets `trace.audit` check a finished answer by looking for
the digits in it.
"""

from __future__ import annotations

import json

from . import enzymes as ez
from .design import Design, Issue
from .seqio import revcomp

LEVEL_MARK = {"error": "FAIL", "warning": "WARN", "note": "note"}


def _rule(title: str = "", width: int = 74) -> str:
    if not title:
        return "-" * width
    return f"-- {title} " + "-" * max(width - len(title) - 4, 0)


def _oligo_block(o, label: str) -> list[str]:
    lines = [
        f"  {label}  {o.name}",
        f"        {o.formatted()}",
        f"        full length   {len(o)} nt",
        f"        annealing     {o.anneal}  ({len(o.anneal)} nt)",
        f"        Tm annealing  {o.tm_anneal:5.1f} C   <- sets the annealing temperature",
        f"        Tm full       {o.tm_full:5.1f} C   (cycle 3 onward, once the tail is copied)",
        f"        GC            {o.gc * 100:5.1f} %  (annealing region)",
        f"        3' dG         {o.three_prime_dg:5.1f} kcal/mol   GC clamp {o.gc_clamp}/5",
    ]
    return lines


def diagnostic_digest(d: Design) -> list[tuple[str, list[int]]]:
    """Fragment sizes from cutting the finished construct."""
    out: list[tuple[str, list[int]]] = []
    five, three = d.pair.five, d.pair.three
    seq = d.construct
    combos = [
        (five.name, [five]),
        (three.name, [three]),
        (f"{five.name} + {three.name}", [five, three]),
    ]
    for label, enzs in combos:
        cuts: list[int] = []
        for e in enzs:
            cuts += [p + e.top for p in ez.find_sites(seq, e.site, circular=True)]
        cuts.sort()
        if not cuts:
            continue
        sizes = [
            (cuts[(i + 1) % len(cuts)] - cuts[i]) % len(seq) for i in range(len(cuts))
        ]
        sizes = [s if s else len(seq) for s in sizes]
        out.append((label, sorted(sizes, reverse=True)))
    return out


def render(d: Design) -> str:
    o = d.options
    lines: list[str] = []
    add = lines.append

    add(_rule())
    add(f"  {d.insert.name}  ->  {d.vector.name}")
    add(_rule())
    add("")

    add(_rule("strategy"))
    add(f"  cloning        restriction + ligation, directional")
    add(f"  mode           {o.mode}")
    add(
        f"  5' enzyme      {d.pair.five.describe()}   "
        f"vector position {d.pair.pos_five + 1}"
    )
    add(
        f"  3' enzyme      {d.pair.three.describe()}   "
        f"vector position {d.pair.pos_three + 1}"
    )
    for enz in (d.pair.five, d.pair.three):
        iso = ez.isoschizomers(enz)
        if iso:
            add(f"                 {enz.name} isoschizomers: {', '.join(iso)}")
    add(f"  excised        {d.pair.excision} bp of vector")
    add(f"  insert         {len(d.core)} bp amplified from the template")
    add("")

    add(_rule("primers"))
    lines += _oligo_block(d.forward, "FWD")
    add("")
    lines += _oligo_block(d.reverse, "REV")
    add("")
    add(f"  Tm difference  {abs(d.forward.tm_anneal - d.reverse.tm_anneal):.1f} C")
    add(f"  suggested Ta   {d.annealing_temp:.1f} C  (lower annealing Tm minus 5)")
    add(f"  gradient       {d.annealing_temp - 4:.0f} - {d.annealing_temp + 4:.0f} C")
    add("")

    add(_rule("expected products"))
    add(f"  PCR amplicon         {len(d.amplicon)} bp")
    add(f"  after double digest  {len(d.insert_fragment)} bp insert fragment")
    add(f"  vector backbone      {len(d.backbone)} bp")
    add(f"  final construct      {len(d.construct)} bp")
    add("")
    add("  diagnostic digest of the finished construct:")
    for label, sizes in diagnostic_digest(d):
        shown = ", ".join(f"{s} bp" for s in sizes)
        add(f"    {label:<24} {shown}")
    add("")

    if d.alternatives:
        add(_rule("alternative enzyme pairs"))
        for alt in d.alternatives:
            note = f"  ({'; '.join(alt.reasons)})" if alt.reasons else ""
            add(
                f"  {alt.label:<18} excises {alt.excision} bp"
                f"{note}"
            )
        add("")

    add(_rule("checks"))
    groups = {"error": [], "warning": [], "note": []}
    for issue in d.issues:
        groups[issue.level].append(issue.text)
    if not any(groups.values()):
        add("  all checks passed")
    for level in ("error", "warning", "note"):
        for text in groups[level]:
            add(f"  {LEVEL_MARK[level]:>5}  {text}")
    add("")

    add(_rule("ordering"))
    add(f"  {d.forward.name}\t{d.forward.sequence}")
    add(f"  {d.reverse.name}\t{d.reverse.sequence}")
    add(_rule())
    return "\n".join(lines)


def _issue_block(issues, indent: str = "  ") -> list[str]:
    """Issues grouped worst-first, in the same shape used elsewhere."""
    lines: list[str] = []
    for level in ("error", "warning", "note"):
        for i in issues:
            if i.level == level:
                lines.append(f"{indent}{LEVEL_MARK[level]:>5}  {i.text}")
    return lines


def _order_block(primers) -> list[str]:
    """The oligos that actually have to be bought."""
    lines: list[str] = []
    for p in primers:
        lines.append(
            f"    {p.name:<10} 5'-{p.sequence}-3'   "
            f"{len(p.sequence):2} nt  Tm {p.tm:4.1f}"
        )
    return lines


def render_sequencing(plan) -> str:
    """The Sanger panel: which reads, from where, and what they cover."""
    g = plan.geometry
    n = len(plan.construct.seq)
    t_start, t_end = plan.target
    span = (t_end - t_start) % n or n
    lo, hi = plan.region

    lines = [_rule("sequencing the construct")]
    lines.append(f"  construct      {n} bp, circular")
    lines.append(
        f"  to verify      {lo % n + 1}-{(hi - 1) % n + 1}   "
        f"({span} bp insert plus {(hi - lo - span) // 2} bp either side, so "
        "the reads"
    )
    lines.append(
        "                 carry into vector sequence and the junctions are "
        "actually read)"
    )
    lines.append(
        f"  read model     bases 1-{g.dead_zone} after the primer discarded, "
        f"usable to {g.usable},"
    )
    lines.append(
        f"                 consecutive reads overlapping by {g.overlap} bp"
    )
    to_order = plan.to_order
    free = len(plan.primers) - len(to_order)
    lines.append(
        f"  reads          {len(plan.primers)}  "
        f"({len(to_order)} primer(s) to order"
        + (f", {free} on universal primers)" if free else ")")
    )
    lines.append("")

    for label, group in (("forward", plan.forward), ("reverse", plan.reverse)):
        if not group:
            continue
        lines.append(f"  {label} reads, in the order they walk the construct:")
        for p in group:
            lines.append(f"    {p.describe(g)}")
        lines.append("")

    if plan.gaps:
        lines.append("  coverage       INCOMPLETE")
        for a, b in plan.gaps:
            lines.append(f"    gap {a % n + 1}-{(b - 1) % n + 1}  ({b - a} bp)")
    else:
        strands = "both strands" if plan.reverse else "one strand"
        lines.append(
            f"  coverage       contiguous across the whole region, {strands}"
        )
    lines.append("")

    if to_order:
        lines.append("  primers to order:")
        lines.extend(_order_block(to_order))
        lines.append("")

    lines.extend(_issue_block(plan.issues))
    return "\n".join(lines)


def render_colony(plan) -> str:
    """The colony screen: which pairs, what each band means."""
    p = plan.polymerase
    lines = [_rule("colony PCR screen")]
    lines.append(f"  polymerase     {p.name}")
    lines.append("")

    if not plan.assays:
        lines.append("  no usable primer pair could be placed.")
        lines.append("")

    for assay in plan.assays:
        lines.append(f"  {assay.name}")
        lines.append(f"    asks         {assay.question}")
        lines.append(
            f"    primers      {assay.forward.name} "
            f"(3' end {assay.forward.site + 1}) x {assay.reverse.name} "
            f"(3' end {assay.reverse.site + 1})"
        )
        rec = (
            f"{assay.recombinant_bp} bp"
            if assay.recombinant_bp is not None
            else "no product"
        )
        lines.append(f"    clone        {rec}")
        if not assay.parental_checked:
            parental = "not checked"
        elif assay.parental_bp is None:
            parental = "no product  <- a band can only come from a real clone"
        else:
            parental = f"{assay.parental_bp} bp"
        lines.append(f"    parental     {parental}")
        lines.extend(_issue_block(assay.issues, indent="    "))
        lines.append("")

    ordered = []
    seen = set()
    for assay in plan.assays:
        for primer in (assay.forward, assay.reverse):
            if primer.name not in seen:
                seen.add(primer.name)
                ordered.append(primer)
    if ordered:
        lines.append("  primers to order:")
        lines.extend(_order_block(ordered))
        lines.append("")

    lines.extend(_issue_block(plan.issues))
    return "\n".join(lines)

def _wrap(text: str, width: int) -> list[str]:
    """Wrap on spaces only, never inside a word.

    `textwrap` breaks hyphenated words by default, which turns
    "machine-readable" into "machine-" and "readable" on two lines.
    That looks harmless and is not: the audit in `trace` checks that
    every issue's text survives into the answer by collapsing
    whitespace and looking for it, and a word split at a hyphen cannot
    be put back by collapsing whitespace.  An issue whose text happened
    to have a hyphen near the wrap column would be reported as dropped,
    and `orchestrate.assemble` would refuse to print a perfectly good
    answer.  Long tokens -- URLs, mostly -- overrun the margin instead
    of being cut, for the same reason and because a URL broken across
    two lines cannot be clicked or copied.
    """
    import textwrap

    return textwrap.wrap(
        text, width, break_on_hyphens=False, break_long_words=False
    ) or [""]


def verification_dict(sequencing=None, colony=None) -> dict:
    """Machine-readable form of the two verification plans."""
    out: dict = {}
    if sequencing is not None:
        g = sequencing.geometry
        n = len(sequencing.construct.seq)
        out["sequencing"] = {
            "read_model": {
                "dead_zone": g.dead_zone,
                "usable": g.usable,
                "overlap": g.overlap,
            },
            "region": [
                sequencing.region[0] % n + 1,
                (sequencing.region[1] - 1) % n + 1,
            ],
            "primers": [
                {
                    "name": p.name,
                    "sequence": p.sequence,
                    "strand": "forward" if p.strand > 0 else "reverse",
                    "three_prime": p.site + 1,
                    "tm": round(p.tm, 2),
                    "universal": p.universal or None,
                    "reads": [
                        p.read(g)[0] % n + 1,
                        p.read(g)[1] % n + 1,
                    ],
                }
                for p in sequencing.primers
            ],
            "gaps": [[a % n + 1, (b - 1) % n + 1] for a, b in sequencing.gaps],
            "repeats": [
                {
                    "length": r.length,
                    "first": r.first + 1,
                    "second": r.second + 1,
                    "inverted": r.inverted,
                }
                for r in sequencing.repeats
            ],
            "issues": [
                {"level": i.level, "text": i.text} for i in sequencing.issues
            ],
        }
    if colony is not None:
        out["colony_pcr"] = {
            "polymerase": colony.polymerase.name,
            "assays": [
                {
                    "name": a.name,
                    "forward": {
                        "name": a.forward.name, "sequence": a.forward.sequence
                    },
                    "reverse": {
                        "name": a.reverse.name, "sequence": a.reverse.sequence
                    },
                    "recombinant_bp": a.recombinant_bp,
                    "parental_bp": a.parental_bp,
                    "parental_checked": a.parental_checked,
                    "spans_junction": a.spans_junction,
                    "discriminates": a.discriminates,
                    "issues": [
                        {"level": i.level, "text": i.text} for i in a.issues
                    ],
                }
                for a in colony.assays
            ],
            "issues": [
                {"level": i.level, "text": i.text} for i in colony.issues
            ],
        }
    return out


def to_dict(d: Design) -> dict:
    return {
        "insert": {"name": d.insert.name, "length": len(d.insert)},
        "vector": {
            "name": d.vector.name,
            "length": len(d.vector),
            "circular": d.vector.circular,
        },
        "strategy": {
            "mode": d.options.mode,
            "five_enzyme": d.pair.five.name,
            "three_enzyme": d.pair.three.name,
            "five_position": d.pair.pos_five + 1,
            "three_position": d.pair.pos_three + 1,
            "excised_bp": d.pair.excision,
        },
        "primers": {
            o.name: {
                "sequence": o.sequence,
                "length": len(o),
                "spacer": o.spacer,
                "site": o.site,
                "pad": o.pad,
                "annealing": o.anneal,
                "tm_annealing": round(o.tm_anneal, 2),
                "tm_full": round(o.tm_full, 2),
                "gc_annealing": round(o.gc, 4),
                "three_prime_dg": round(o.three_prime_dg, 2),
            }
            for o in (d.forward, d.reverse)
        },
        "annealing_temperature": round(d.annealing_temp, 1),
        "products": {
            "amplicon_bp": len(d.amplicon),
            "insert_fragment_bp": len(d.insert_fragment),
            "backbone_bp": len(d.backbone),
            "construct_bp": len(d.construct),
        },
        "diagnostic_digest": {k: v for k, v in diagnostic_digest(d)},
        "alternatives": [
            {"pair": a.label, "excised_bp": a.excision} for a in d.alternatives
        ],
        "issues": [{"level": i.level, "text": i.text} for i in d.issues],
    }


def render_json(d: Design) -> str:
    return json.dumps(to_dict(d), indent=2)


def render_source(assessment) -> str:
    """What the gene's origin costs in E. coli."""
    a = assessment
    lines = [_rule("the gene and where it came from")]
    lines.append(f"  gene           {a.insert.name}, {len(a.insert)} bp")
    lines.append(f"  source         {a.origin.label}")
    lines.append(
        f"  reading frame  "
        + (
            "clean: ATG to stop, in frame, uninterrupted"
            if a.orf.is_clean
            else "NOT clean - " + "; ".join(a.orf.faults())
        )
    )
    from .source import ECOLI_BASELINE, RARE_CODONS

    lines.append(
        f"  rare codons    {a.codons.rare} of {a.codons.total} "
        f"({a.codons.fraction:.1%}); E. coli's own genes sit near "
        f"{ECOLI_BASELINE:.1%}"
    )
    if a.codons.counts:
        rows = sorted(a.codons.counts.items(), key=lambda kv: -kv[1])
        lines.append("")
        lines.append("    codon  aa   count   per 1000 in E. coli   pRARE")
        for codon, count in rows:
            aa, freq, supplied = RARE_CODONS[codon]
            lines.append(
                f"    {codon}    {aa}  {count:>5}   {freq:>17.1f}   "
                + ("yes" if supplied else "no")
            )
    lines.append("")
    lines.append(
        f"  strain hint    "
        + (
            "a tRNA-supplementing strain such as Rosetta is worth using"
            if a.wants_rare_codon_strain
            else "the rare-codon load does not call for a special strain"
        )
    )
    lines.append("")
    for issue in a.issues:
        for index, chunk in enumerate(_wrap(issue.text, 66)):
            mark = LEVEL_MARK[issue.level] if index == 0 else ""
            lines.append(f"  {mark:>5}  {chunk}")
    return "\n".join(lines).rstrip()



def render_expression(plan) -> str:
    """The protein, what induces it, and which strain can make it.

    The panel exists because the DNA report answers a different question.
    A design can be perfect -- right enzymes, matched primers, frame
    carried through -- and still describe a construct that expresses
    nothing, because the strain cannot transcribe the promoter or because
    the tag everyone is counting on sits past a stop codon.
    """
    spec = plan.spec
    hosts = plan.hosts
    product = plan.product

    lines = [_rule("expression in E. coli")]
    if spec is not None:
        lines.append(f"  vector         {spec.name}")
        lines.append(f"  promoter       {spec.promoter}")
        lines.append(f"  induction      {spec.induction}")
        lines.append(f"  selection      {spec.antibiotic}")
        lines.append(f"  purification   {spec.purification}")
    else:
        view = plan.view
        promoter = view.promoter.name if view.promoter is not None else "not annotated"
        lines.append(f"  promoter       {promoter}")
        lines.append(f"  cassette       reads "
                     + ("along the map" if not view.flipped else "against the map"))
    lines.append(f"  cloning host   {hosts.cloning.describe()}")
    if hosts.expression is not None:
        lines.append(f"  expression     {hosts.expression.describe()}")
    if hosts.antibiotics:
        lines.append(f"  plates need    {', '.join(hosts.antibiotics)}")
    lines.append("")

    if product is not None:
        lines.append("  the protein this construct encodes:")
        lines.append(f"    length       {len(product)} residues")
        lines.append(f"    mass         {product.mass_kda:.1f} kDa")
        if product.extinction:
            lines.append(
                f"    A280         {product.a280_for_1mg_ml:.2f} for 1 mg/mL, "
                f"1 cm (e = {product.extinction:.0f} M-1 cm-1)"
            )
        else:
            lines.append("    A280         no W, Y or C: not measurable at 280 nm")
        if product.leader:
            lines.append(f"    N terminus   {product.leader}")
            lines.append(f"                 ({len(product.leader)} residues from "
                         "the vector, ahead of your sequence)")
        else:
            lines.append("    N terminus   nothing added; your own start codon")
        if product.trailer:
            lines.append(f"    C terminus   {product.trailer} "
                         f"({len(product.trailer)} residues from the vector)")
        else:
            lines.append("    C terminus   nothing added")
        if product.stop_codon:
            lines.append(f"    stops at     {product.stop_codon}")
        lines.append("")
        peptide = product.peptide
        lines.append("  full sequence, N to C:")
        for i in range(0, len(peptide), 60):
            lines.append(f"    {i + 1:>5}  {peptide[i:i + 60]}")
        lines.append("")

    lines.extend(_issue_block(plan.issues))
    return "\n".join(lines).rstrip()


def render_shortlist(usable, rejected) -> str:
    """Every bundled vector, judged by designing into this insert.

    Ordered by what the design produced, not by which vector is better
    known: a plasmid whose only useful enzymes cut inside this particular
    gene is not a candidate for this gene.
    """
    lines = [_rule("vectors, ranked by designing into each one")]
    if not usable and not rejected:
        lines.append("  no bundled vectors are available.")
        return "\n".join(lines)

    if usable:
        lines.append(f"  {'vector':<14}{'enzymes':<15}{'mode':<10}"
                     f"{'product':<17}{'tag':<6}warnings")
        for c in usable:
            product = ""
            if c.product_residues is not None:
                product = f"{c.product_residues} aa, {c.product_kda:.0f} kDa"
            lines.append(
                f"  {c.spec.name:<14}{c.pair:<15}{c.mode:<10}"
                f"{product:<17}{'yes' if c.tagged else 'no':<6}{c.warnings}"
            )
        lines.append("")
        for c in usable:
            lines.append(f"  {c.spec.name}")
            lines.append(f"    N terminus   {c.spec.n_terminal}")
            lines.append(f"    C terminus   {c.spec.c_terminal}")
            for chunk in _wrap(c.spec.note, 64):
                lines.append(f"    {chunk}")
            lines.append("")

    if rejected:
        lines.append("  rejected for this insert:")
        for c in rejected:
            for index, chunk in enumerate(_wrap(c.refusal, 60)):
                head = f"{c.spec.name}: " if index == 0 else ""
                lines.append(f"    {head}{chunk}")
        lines.append("")
    return "\n".join(lines).rstrip()
