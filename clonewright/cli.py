"""The command line: six subcommands, three exit codes.

    0  a design that succeeded
    1  a design that was refused
    2  a bad invocation, or a file that could not be read

A refusal is a result, not a crash, and it exits 1 rather than 0 so a
script can tell the two apart without parsing the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import agent as agent_mod
from . import cassette, construct, design, interactive, orchestrate
from . import planner, report
from . import ncbi, seqio, vectors
from .seqio import SequenceError
from .trace import InvariantError

EPILOG = """\
examples:
  clonewright design SOD1.fasta --vector pET-28a
  clonewright design SOD1.fasta                     # it picks the vector
  clonewright ask "clone SOD1.fasta into pET-28a, it is a human gene"
  clonewright design SOD1.fasta --vector pET-28a --origin human
  clonewright design "https://www.ncbi.nlm.nih.gov/nuccore/NM_007117.5"
  clonewright design NC_000003.12 --cds TRH --vector pGEX-4T-1
  clonewright design gene.fa --vector ./my-pET-28a.dna --json
  clonewright compare SOD1.fasta
  clonewright annotate unannotated-vector.fa -o annotated.gb
  clonewright                                       # asks what it needs
  clonewright vectors
  clonewright tools
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clonewright",
        description=(
            "Design restriction-cloning primers for expressing a gene in "
            "E. coli, checked against the vector and the insert rather than "
            "assumed to work."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument("--version", action="version",
                   version=f"clonewright {__version__}")
    subs = p.add_subparsers(dest="command")

    d = subs.add_parser(
        "design", help="design primers for one gene into one vector",
        description="The whole pipeline: enzyme pair, primers, reading "
                    "frame, the protein, the strain, and the primers that "
                    "prove the construct.",
    )
    d.add_argument("insert", help="the gene: an NCBI link or accession, a "
                                  "FASTA/GenBank/.dna file, or a bare "
                                  "sequence")
    d.add_argument("--vector", default="", metavar="NAME|PATH",
                   help="a bundled vector by name, or any map by path. "
                        "Omit it and every bundled vector is tried and the "
                        "best-ranked one used")
    d.add_argument("--mode", default="auto",
                   choices=("auto", "native", "nfusion", "cfusion"),
                   help="auto reads the vector and decides (default)")
    d.add_argument("--cds", default="", metavar="GENE",
                   help="which coding sequence to take, when an NCBI "
                        "region holds more than one gene")
    d.add_argument("--offline", action="store_true",
                   help="never reach the network: read NCBI records from "
                        "the local cache only")
    d.add_argument("--origin", default="", metavar="ORGANISM",
                   help="where the gene came from, e.g. human, yeast, "
                        "e.coli: decides the intron and rare-codon checks")
    d.add_argument("--cloning-strain", default="dh5alpha", metavar="NAME")
    d.add_argument("--expression-strain", default="bl21de3", metavar="NAME")
    d.add_argument("--no-verify", action="store_true",
                   help="skip the colony-PCR and sequencing panels")
    d.add_argument("--compare", action="store_true",
                   help="also show every bundled vector ranked")
    d.add_argument("--json", action="store_true",
                   help="machine-readable output instead of the report")
    d.add_argument("--emit-construct", metavar="FILE",
                   help="write the finished plasmid as GenBank")

    c = subs.add_parser(
        "compare", help="rank every bundled vector for one gene",
        description="Design into every bundled vector and report what "
                    "actually happened in each.",
    )
    c.add_argument("insert")
    c.add_argument("--mode", default="auto",
                   choices=("auto", "native", "nfusion", "cfusion"))

    a = subs.add_parser(
        "annotate", help="derive a bacterial vector's cassette from its sequence",
        description="For a vector that arrived as bare sequence. Finds the "
                    "promoter, operator, ribosome binding site, tags, "
                    "protease sites, polylinker and open reading frames, and "
                    "writes a GenBank file. Every search can fail, and "
                    "failing means no feature rather than a wrong one.",
    )
    a.add_argument("vector")
    a.add_argument("-o", "--out", metavar="FILE",
                   help="where to write the GenBank (default: stdout)")
    a.add_argument("--partner", default="", metavar="NAME",
                   help="name for the N-terminal fusion partner, if the "
                        "vector has one, e.g. GST")
    a.add_argument("--promoter", default="", metavar="NAME",
                   help="name for a sigma-70 promoter, e.g. 'tac promoter'")
    a.add_argument("--min-orf", type=int, default=300, metavar="BP")

    k = subs.add_parser(
        "ask", help="say what you want in a sentence",
        description="Reads the request out of a sentence and runs the same "
                    "pipeline. Patterns do the reading by default, with no "
                    "model and no network; --model asks Claude instead, for "
                    "the sentences patterns do not cover. Either way the "
                    "result is validated against the catalog, the mode list "
                    "and the strain table before any tool sees it -- and a "
                    "planner cannot put a number into the answer, because "
                    "there is no numeric field for it to put one in.",
    )
    k.add_argument("question", help="e.g. \"clone SOD1.fasta into pET-28a, "
                                    "it's a human gene\"")
    k.add_argument("--model", action="store_true",
                   help="use Claude to read the sentence "
                        "(needs clonewright[agent] and a credential)")
    k.add_argument("--dry-run", action="store_true",
                   help="print the request that was understood, and stop")

    subs.add_parser(
        "interactive", help="ask for what is needed, one question at a time",
        description="The same pipeline, driven by four questions instead of "
                    "flags. This is what bare `clonewright` runs when there "
                    "is a terminal to ask on.",
    )
    subs.add_parser("vectors", help="list the bundled vectors")
    subs.add_parser("tools", help="list the tool registry the agent drives")
    return p


# ---------------------------------------------------------------------------


def _request(args) -> orchestrate.Request:
    return orchestrate.Request(
        insert=args.insert,
        vector=getattr(args, "vector", ""),
        mode=args.mode,
        origin=getattr(args, "origin", ""),
        cloning_strain=getattr(args, "cloning_strain", "dh5alpha"),
        expression_strain=getattr(args, "expression_strain", "bl21de3"),
        verify=not getattr(args, "no_verify", False),
        compare=getattr(args, "compare", False),
        cds_gene=getattr(args, "cds", ""),
        allow_network=not getattr(args, "offline", False),
    )


def _json_out(out: orchestrate.Outcome) -> str:
    payload: dict = {
        "refused": out.refused,
        "refusal": out.refusal,
        "vector": getattr(out.spec, "name", out.request.vector),
        "issues": [
            {"level": i.level, "text": i.text} for i in out.trace.issues
        ],
        "tool_calls": [
            {"tool": c.tool, "layer": c.layer, "outcome": c.outcome}
            for c in out.trace.calls
        ],
    }
    if out.design is not None:
        payload["design"] = report.to_dict(out.design)
    if out.plan is not None and out.plan.product is not None:
        product = out.plan.product
        payload["product"] = {
            "residues": len(product),
            "mass_kda": round(product.mass_kda, 2),
            "extinction_280": product.extinction,
            "a280_1mg_ml": round(product.a280_for_1mg_ml, 3),
            "n_terminal_addition": product.leader,
            "c_terminal_addition": product.trailer,
            "sequence": product.peptide,
        }
    if out.candidates:
        payload["vectors_ranked"] = [
            {
                "vector": c.spec.name,
                "enzymes": c.pair,
                "mode": c.mode,
                "product_residues": c.product_residues,
                "tagged": c.tagged,
                "warnings": c.warnings,
            }
            for c in out.candidates
        ]
    return json.dumps(payload, indent=2)


def cmd_design(args) -> int:
    try:
        out = orchestrate.run(_request(args))
    except SequenceError as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2
    except (vectors.VectorError, ncbi.FetchError) as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2
    except InvariantError as exc:
        # The audit refused the finished answer.  Printing it anyway would
        # defeat the point of having checked it, so nothing is printed but
        # the reason -- and it exits 2, because this is a fault in
        # clonewright, not in the user's cloning.
        print(f"clonewright: the answer failed its own audit ({exc})",
              file=sys.stderr)
        return 2

    print(_json_out(out) if args.json else out.answer)

    if args.emit_construct and out.design is not None:
        d = out.design
        record = construct.assemble(
            d, name=f"{d.insert.name}_in_{getattr(out.spec, 'name', 'vector')}"
        )
        Path(args.emit_construct).write_text(
            seqio.write_genbank(
                record,
                definition=f"{d.insert.name} cloned into "
                           f"{getattr(out.spec, 'name', 'vector')} with "
                           f"{d.pair.label} by clonewright {__version__}",
            ),
            encoding="utf-8",
        )
        print(f"\nconstruct written to {args.emit_construct}",
              file=sys.stderr)

    return 1 if out.refused else 0


def cmd_compare(args) -> int:
    try:
        insert = seqio.load_sequence(args.insert)
    except SequenceError as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2
    usable, rejected = vectors.shortlist(
        insert, opts=design.Options(mode=args.mode)
    )
    print(report.render_shortlist(usable, rejected))
    return 0 if usable else 1


def cmd_annotate(args) -> int:
    try:
        record = seqio.load(args.vector)
    except SequenceError as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2
    hints = cassette.Hints(partner=args.partner, promoter=args.promoter)
    annotated = cassette.annotate(record, hints=hints,
                                  min_orf_bp=args.min_orf)
    text = seqio.write_genbank(
        annotated,
        definition=f"{annotated.name}, cassette derived from sequence by "
                   f"clonewright {__version__}",
    )
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{len(annotated.features)} feature(s) written to {args.out}")
        for f in annotated.features:
            print(f"  {f.describe()}")
    else:
        print(text, end="")
    return 0


def cmd_ask(args) -> int:
    chosen = planner.for_question(args.question, use_model=args.model)
    try:
        request = chosen.route(orchestrate.Request(insert=""))
    except planner.PlannerError as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2

    chose_for_you = "it picks one" if not request.vector else request.vector
    print(f"understood: clone {request.insert} into {chose_for_you}")
    if request.origin:
        print(f"            gene from {request.origin}")
    print(f"            mode {request.mode}, "
          f"verification {'on' if request.verify else 'off'}"
          + (", every vector compared" if request.compare else "") + "\n")
    if args.dry_run:
        return 0

    try:
        out = orchestrate.run(request)
    except (SequenceError, vectors.VectorError, ncbi.FetchError) as exc:
        print(f"clonewright: {exc}", file=sys.stderr)
        return 2
    except InvariantError as exc:
        print(f"clonewright: the answer failed its own audit ({exc})",
              file=sys.stderr)
        return 2
    print(out.answer)
    return 1 if out.refused else 0


def cmd_vectors(_args) -> int:
    print("bundled E. coli expression vectors:\n")
    for name, spec in vectors.CATALOG.items():
        mark = " " if spec.path.exists() else "!"
        print(f"{mark} {name:<12} {spec.promoter:<8} {spec.antibiotic}")
        print(f"    N: {spec.n_terminal}")
        print(f"    C: {spec.c_terminal}")
        if spec.requires:
            print(f"    needs: {spec.requires}")
        print()
    missing = [n for n, s in vectors.CATALOG.items() if not s.path.exists()]
    if missing:
        print(f"! map file missing for: {', '.join(missing)}")
        print("  build it with tools/build_vector_maps.py")
    print("Any other vector can be given to --vector as a path to a "
          "GenBank,\nSnapGene .dna or FASTA map.")
    return 0


def cmd_tools(_args) -> int:
    print("the registry the agent drives, by layer:\n")
    print(agent_mod.catalogue())
    print(
        "\nLayer 0 acquires, 1 computes, 4 renders. A model may call any of "
        "\nlayer 1 and may not reimplement, approximate or second-guess any "
        "of it:\nevery number in an answer originates there."
    )
    return 0


COMMANDS = {
    "design": cmd_design,
    "ask": cmd_ask,
    "interactive": lambda _args: interactive.run(),
    "compare": cmd_compare,
    "annotate": cmd_annotate,
    "vectors": cmd_vectors,
    "tools": cmd_tools,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        # Bare `clonewright` starts the interview -- that is what somebody
        # who has just installed it types, and a wall of flags is a poor
        # answer to it.  With no terminal on the other end there is nobody
        # to interview, and a blocking prompt in a pipe or a CI step is
        # worse than help text, so that case keeps the help.
        if interactive.available():
            return interactive.run()
        parser.print_help()
        return 2
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
