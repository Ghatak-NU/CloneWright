"""Build the bundled vector maps in `clonewright/data/vectors/`.

The maps this package ships carry **derived** annotations.  Point this
script at a vector map you already have -- a SnapGene `.dna` from the
vendor, a GenBank file, a FASTA of the plasmid -- and it throws the
existing feature table away, finds the cassette in the sequence with
`clonewright.cassette`, and writes a GenBank file containing the sequence
plus what it found.

Why it works this way.  A vendor's feature table is their annotation of
their plasmid, and copying one into a public repository redistributes it.
The sequence of pET-28a is a fact about a molecule thousands of labs use;
where its ribosome binding site sits is something you can measure from
those bases, and this script measures it.  The only thing it is told
rather than shown is what an open reading frame *is* -- that the 816 bp
ORF confers kanamycin resistance -- and those names are published
properties of the vectors, matched to ORFs by length.

Everything in a built map is therefore either sequence, arithmetic, or a
name from the table below.  Run it and diff the output: that is the point
of shipping plain text.

    py tools/build_vector_maps.py --source-dir "/path/to/your/vector/maps"
    py tools/build_vector_maps.py --only pET-28a --source ~/maps/pET-28a.dna
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clonewright import cassette, seqio  # noqa: E402

OUT_DIR = ROOT / "clonewright" / "data" / "vectors"

# Coding-sequence lengths in base pairs, used to put a name to an open
# reading frame.  Published properties of the genes, not of these
# plasmids: KanR (aph(3')-Ia) is 816 bp with its stop, AmpR (bla) 861,
# lacI 1083, rop 192.  An ORF more than 5% away from one of these keeps
# its measured length as its name instead.
COMMON_ORFS = {"KanR": 816, "AmpR": 861, "lacI": 1083, "rop": 192}

VECTORS: dict[str, dict] = {
    "pET-28a": {
        "definition": (
            "pET-28a(+), T7 expression vector with N-terminal 6xHis-thrombin "
            "and C-terminal 6xHis. Annotations derived from sequence by "
            "clonewright.cassette"
        ),
        "filenames": ("pet-28a", "pet28a"),
        "hints": {"orfs": COMMON_ORFS},
    },
    "pET-21b": {
        "definition": (
            "pET-21b(+), T7 expression vector with C-terminal 6xHis and no "
            "N-terminal tag. Annotations derived from sequence by "
            "clonewright.cassette"
        ),
        "filenames": ("pet-21b", "pet21b"),
        "hints": {"orfs": COMMON_ORFS},
    },
    "pGEX-4T-1": {
        "definition": (
            "pGEX-4T-1, tac expression vector with an N-terminal GST fusion "
            "partner and a thrombin site. Annotations derived from sequence "
            "by clonewright.cassette"
        ),
        "filenames": ("pgex-4t-1", "pgex4t1"),
        "hints": {
            "orfs": COMMON_ORFS,
            "partner": "GST",
            "promoter": "tac promoter",
        },
    },
    "pMAL-c5X": {
        "definition": (
            "pMAL-c5X, tac expression vector with an N-terminal MBP fusion "
            "partner and a factor Xa site. Annotations derived from sequence "
            "by clonewright.cassette"
        ),
        "filenames": ("pmal-c5x", "pmalc5x"),
        "hints": {
            "orfs": COMMON_ORFS,
            "partner": "MBP",
            "promoter": "tac promoter",
        },
    },
}


def find_source(name: str, source_dir: Path) -> Path | None:
    """A map for `name` in `source_dir`, by any of its spellings.

    The closest name wins, not the first one found.  `pMAL-c5X-His.dna`
    contains `pmal-c5x` as a substring and is a different plasmid -- it
    carries an extra His tag and is 18 bp longer -- so a plain substring
    match silently built the wrong map.
    """
    wanted = VECTORS[name]["filenames"]
    best: tuple[int, Path] | None = None
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in (".dna", ".gb", ".gbk", ".fa", ".fasta"):
            continue
        stem = path.stem.lower().replace("(+)", "").replace("_", "-")
        squashed = stem.replace("-", "")
        for w in wanted:
            if stem == w or squashed == w.replace("-", ""):
                return path
            if w in stem or w.replace("-", "") in squashed:
                # Prefer the shortest match: extra characters mean extra
                # parts in the plasmid's name, which mean a variant of it.
                if best is None or len(stem) < best[0]:
                    best = (len(stem), path)
    return best[1] if best else None


def build(name: str, source: Path) -> Path:
    spec = VECTORS[name]
    loaded = seqio.load(source)
    stripped = seqio.Record(name, loaded.seq, loaded.circular, [])
    hints = cassette.Hints(**spec["hints"])
    annotated = cassette.annotate(stripped, hints=hints)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.gb"
    out.write_text(
        seqio.write_genbank(annotated, definition=spec["definition"]),
        encoding="utf-8",
    )
    kinds = ", ".join(
        f"{f.name}" for f in annotated.features
        if f.kind in ("promoter", "RBS", "misc_feature")
    )
    print(
        f"{name:<12} {len(annotated):>5} bp  "
        f"{len(annotated.features):>2} features  from {source.name}"
    )
    print(f"{'':<12} cassette: {kinds}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-dir", type=Path,
                   help="directory to search for vector map files")
    p.add_argument("--source", type=Path,
                   help="one map file; requires --only")
    p.add_argument("--only", help="build just this vector")
    args = p.parse_args(argv)

    names = [args.only] if args.only else list(VECTORS)
    for name in names:
        if name not in VECTORS:
            print(f"unknown vector {name!r}; known: {', '.join(VECTORS)}",
                  file=sys.stderr)
            return 2

    if args.source is not None:
        if not args.only:
            print("--source needs --only", file=sys.stderr)
            return 2
        build(args.only, args.source)
        return 0

    if args.source_dir is None:
        print("give --source-dir, or --only with --source", file=sys.stderr)
        return 2

    missing: list[str] = []
    for name in names:
        source = find_source(name, args.source_dir)
        if source is None:
            missing.append(name)
            continue
        build(name, source)
    for name in missing:
        print(f"{name:<12} no source map found under {args.source_dir}",
              file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
