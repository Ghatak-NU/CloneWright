"""Reading DNA from FASTA, GenBank, raw text, and SnapGene .dna files.

Three formats, because a cloning tool meets all three: the insert usually
arrives as FASTA, the vector as a SnapGene map someone downloaded from the
vendor, and the vectors this package ships with are GenBank flat files --
plain text, diffable, and readable without SnapGene installed.

SnapGene .dna is a segmented binary format.  Each segment is::

    1 byte   segment type
    4 bytes  payload length, big-endian
    N bytes  payload

Segment type 0 carries the sequence: its first payload byte is a flags
byte whose low bit marks the molecule as circular, and the rest is the
sequence as ASCII.  Segment type 10 carries the feature table as XML.
Parsing the container properly (rather than scraping the longest ACGT run
out of the file) is what lets us read topology and annotations, which the
site arithmetic and the cassette-replacement logic depend on.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
    "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT",
}

UNAMBIGUOUS = frozenset("ACGT")

_COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVNacgtryswkmbdhvn",
    "TGCAYRSWMKVHDBNtgcayrswmkvhdbn",
)


def revcomp(seq: str) -> str:
    """Reverse complement, IUPAC-aware."""
    return seq.translate(_COMPLEMENT)[::-1]


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


# The standard genetic code, in the TCAG ordering the table is
# conventionally written in, so it can be checked against a printed copy
# rather than trusted.  Codon n is _AMINO[n] with T=0, C=1, A=2, G=3.
_BASES = "TCAG"
_AMINO = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
CODON_TABLE = {
    a + b + c: _AMINO[i * 16 + j * 4 + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}


def translate(seq: str, *, to_stop: bool = False) -> str:
    """Nucleotides to one-letter amino acids, reading frame 1.

    A codon containing anything but A, C, G or T becomes `X`: an
    ambiguity code is a real thing to find in a database sequence, and
    guessing which base was meant would put an invented residue into a
    comparison whose whole job is to decide whether two sequences are
    the same.  A trailing partial codon is dropped.
    """
    seq = seq.upper().replace("U", "T")
    out = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        residue = CODON_TABLE.get(seq[i : i + 3], "X")
        if residue == "*" and to_stop:
            break
        out.append(residue)
    return "".join(out)


class SequenceError(ValueError):
    """Raised when input cannot be read as DNA."""


@dataclass(frozen=True)
class Feature:
    """An annotated region, in 0-based half-open coordinates."""

    name: str
    kind: str
    start: int
    end: int
    forward: bool = True

    def __len__(self) -> int:
        return self.end - self.start

    def describe(self) -> str:
        strand = "+" if self.forward else "-"
        return f"{self.name} [{self.kind}] {self.start + 1}-{self.end} ({strand})"


@dataclass
class Record:
    """A DNA molecule plus the topology and annotations that matter here."""

    name: str
    seq: str
    circular: bool = False
    features: list[Feature] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.seq)

    def find_feature(self, name: str) -> Feature | None:
        wanted = name.strip().lower()
        exact = [f for f in self.features if f.name.lower() == wanted]
        if exact:
            return max(exact, key=len)
        partial = [f for f in self.features if wanted in f.name.lower()]
        return max(partial, key=len) if partial else None

    def reverse_complement(self) -> "Record":
        """The same molecule read from the other strand.

        Features come with it: a span at [start, end) measured from the
        left end sits at [n - end, n - start) measured from the right,
        and its sense flips.  Keeping the annotations attached is the
        whole point -- a reverse-complemented sequence with a stale
        feature table is worse than no feature table, because every
        coordinate in it now points somewhere plausible and wrong.
        """
        n = len(self.seq)
        flipped = [
            Feature(f.name, f.kind, n - f.end, n - f.start, not f.forward)
            for f in self.features
        ]
        flipped.sort(key=lambda f: (f.start, f.end))
        return Record(self.name, revcomp(self.seq), self.circular, flipped)

    def window(self, start: int, end: int) -> str:
        """Half-open [start, end) in 0-based coordinates, wrapping if circular."""
        n = len(self.seq)
        if not self.circular:
            return self.seq[start:end]
        start %= n
        end %= n
        if start < end:
            return self.seq[start:end]
        return self.seq[start:] + self.seq[:end]


def _clean(raw: str, *, source: str) -> str:
    seq = re.sub(r"\s+", "", raw).upper()
    if not seq:
        raise SequenceError(f"{source}: no sequence found")
    bad = sorted(set(seq) - set(IUPAC))
    if bad:
        raise SequenceError(
            f"{source}: not DNA, unexpected character(s): {', '.join(bad)}"
        )
    return seq


def parse_fasta(text: str, *, source: str = "<fasta>") -> list[Record]:
    records: list[Record] = []
    name = Path(source).stem
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if chunks:
                records.append(Record(name, _clean("".join(chunks), source=source)))
                chunks = []
            name = line[1:].strip().split()[0] if line[1:].strip() else name
        else:
            chunks.append(line)
    if chunks:
        records.append(Record(name, _clean("".join(chunks), source=source)))
    if not records:
        raise SequenceError(f"{source}: no FASTA records found")
    return records


def _parse_features(xml_text: str) -> list[Feature]:
    """Read the SnapGene feature table.

    Each Feature holds one or more Segments whose `range` is "start-end",
    1-based and inclusive; the feature spans their union.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    features: list[Feature] = []
    for node in root.iter("Feature"):
        bounds: list[tuple[int, int]] = []
        for seg in node.iter("Segment"):
            rng = seg.get("range", "")
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", rng)
            if m:
                bounds.append((int(m.group(1)), int(m.group(2))))
        if not bounds:
            continue
        start = min(b[0] for b in bounds) - 1
        end = max(b[1] for b in bounds)
        features.append(
            Feature(
                name=node.get("name", "unnamed"),
                kind=node.get("type", "misc_feature"),
                start=start,
                end=end,
                forward=node.get("directionality", "1") != "2",
            )
        )
    features.sort(key=lambda f: (f.start, f.end))
    return features


def parse_snapgene(data: bytes, *, source: str = "<snapgene>") -> Record:
    pos = 0
    seq: str | None = None
    circular = False
    features: list[Feature] = []
    while pos + 5 <= len(data):
        seg_type = data[pos]
        seg_len = int.from_bytes(data[pos + 1 : pos + 5], "big")
        payload = data[pos + 5 : pos + 5 + seg_len]
        pos += 5 + seg_len
        if seg_type == 0 and payload:
            circular = bool(payload[0] & 0x01)
            seq = payload[1:].decode("ascii", "ignore")
        elif seg_type == 10 and payload:
            features = _parse_features(payload.decode("utf-8", "ignore"))
    if seq is None:
        raise SequenceError(f"{source}: no sequence segment in SnapGene file")
    return Record(Path(source).stem, _clean(seq, source=source), circular, features)


# A bound may carry a `<` or `>` partial marker, and it may carry it on
# either end of the range: `<1..228`, `18..>228`, `<1..>228` are all real
# GenBank. Leaving the markers out of the pattern does not produce a
# wrong answer, it produces no answer -- the location fails to parse and
# the feature disappears. NCBI writes `18..>228` for a CDS that runs off
# the end of a requested region, which is precisely the case worth
# reporting clearly, so the feature going missing there was the worst
# place for it to happen.
_LOCATION = re.compile(r"[<>]?(\d+)\s*\.\.\s*[<>]?(\d+)")
_PARTIAL = re.compile(r"[<>]")


@dataclass(frozen=True)
class Location:
    """Where a feature sits, keeping its pieces.

    `segments` are 0-based half-open spans in ascending coordinate order,
    one per exon of a joined location.  `start` and `end` are the outer
    bounds, which is all most callers want; the segments matter to exactly
    one caller, and it is the one that matters most -- splicing a coding
    sequence out of genomic DNA. Collapsing `join(1098..1308,1980..2497)`
    to 1098..2497 there would silently include 671 bases of intron.

    `partial` records a `<` or `>` marker. A partial feature still
    occupies the span it names, so the bounds are kept, but the flag is
    the difference between "this is the gene" and "this is as much of the
    gene as the region you asked for contains".
    """

    segments: tuple[tuple[int, int], ...]
    forward: bool = True
    partial: bool = False

    @property
    def start(self) -> int:
        return min(s for s, _ in self.segments)

    @property
    def end(self) -> int:
        return max(e for _, e in self.segments)

    @property
    def spliced_length(self) -> int:
        return sum(e - s for s, e in self.segments)

    def extract(self, seq: str) -> str:
        """The feature's own sequence, exons joined and strand applied.

        GenBank's `complement(join(a,b))` means: concatenate the pieces in
        the order written, then reverse complement the result. Doing it in
        that order rather than reverse-complementing each piece is what
        keeps the exon boundaries in the right places.
        """
        joined = "".join(seq[s:e] for s, e in self.segments)
        return joined if self.forward else revcomp(joined)


def parse_location(raw: str) -> Location | None:
    """Read a GenBank location string.

    Handles the forms that turn up in real records: `1..20`,
    `complement(1..20)`, `join(1098..1308,1980..2497)` for a spliced CDS,
    `join(5300..5369,1..40)` for a plasmid feature crossing the origin,
    `complement(join(...))`, and the `<1..>20` partial markers.
    """
    bounds = _LOCATION.findall(raw)
    forward = "complement" not in raw.lower()
    partial = bool(_PARTIAL.search(raw))
    if not bounds:
        single = re.fullmatch(r"\s*[<>]?(\d+)[<>]?\s*", raw)
        if not single:
            return None
        pos = int(single.group(1))
        return Location(((pos - 1, pos),), forward, partial)
    segments = tuple(sorted((int(a) - 1, int(b)) for a, b in bounds))
    return Location(segments, forward, partial)


def _parse_location(raw: str) -> tuple[int, int, bool] | None:
    """The outer bounds of a location, for the feature table.

    `Feature` carries one span, because that is what every question about
    a plasmid map needs. Splicing asks a different question and calls
    `parse_location` directly.
    """
    location = parse_location(raw)
    if location is None:
        return None
    return location.start, location.end, location.forward


def parse_genbank(text: str, *, source: str = "<genbank>") -> Record:
    """Read a GenBank flat file, feature table included.

    The feature table is the point.  Every host, promoter, frame and
    polylinker decision downstream is made by reading feature names, so a
    parser that returned the sequence alone would hand the design code a
    vector it cannot reason about -- and it would do it quietly, by
    producing a `Record` that looks fine and has nothing in it.
    """
    lines = text.splitlines()
    name = Path(source).stem
    circular = False
    for line in lines:
        if line.startswith("LOCUS"):
            parts = line.split()
            if len(parts) > 1:
                name = parts[1]
            circular = "circular" in line.lower()
            break

    features: list[Feature] = []
    key = location = ""
    label = gene = product = note = ""

    def flush() -> None:
        nonlocal key, location, label, gene, product, note
        if key and location:
            span = _parse_location(location)
            if span is not None:
                start, end, forward = span
                features.append(Feature(
                    name=label or gene or product or note[:40] or key,
                    kind=key,
                    start=start,
                    end=end,
                    forward=forward,
                ))
        key = location = label = gene = product = note = ""

    in_features = False
    for line in lines:
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if not in_features:
            continue
        if line.startswith("ORIGIN") or line.startswith("//"):
            flush()
            break
        if not line.startswith(" "):
            # Any other column-0 keyword ends the table.
            flush()
            in_features = False
            continue
        stripped = line.strip()
        # A feature key sits at column 5; its qualifiers at column 21.
        if len(line) > 5 and line[5] != " " and not stripped.startswith("/"):
            flush()
            parts = stripped.split(maxsplit=1)
            key = parts[0]
            location = parts[1] if len(parts) > 1 else ""
        elif stripped.startswith("/"):
            qualifier, _, value = stripped[1:].partition("=")
            value = value.strip().strip('"')
            if qualifier == "label" and not label:
                label = value
            elif qualifier in ("gene", "standard_name") and not gene:
                gene = value
            elif qualifier == "product" and not product:
                product = value
            elif qualifier == "note" and not note:
                note = value
        elif key and not location:
            location = stripped
        elif key and location and not stripped.startswith("/"):
            # A location wrapped onto a second line.
            location += stripped

    if "ORIGIN" not in text:
        raise SequenceError(f"{source}: no ORIGIN block; this is not GenBank")
    body = text.split("ORIGIN", 1)[1].split("//")[0]
    seq = _clean(re.sub(r"[\d\s]+", "", body), source=source)
    features.sort(key=lambda f: (f.start, f.end))
    return Record(name, seq, circular, features)


def write_genbank(record: Record, *, definition: str = "") -> str:
    """A Record as a GenBank flat file, openable in SnapGene or Benchling.

    The counterpart to `parse_genbank`, and the reason a finished construct
    can leave this package as something a person can look at rather than as
    a wall of bases.  Features are written with `/label`, which is what
    both of those programs read back.
    """
    n = len(record.seq)
    topology = "circular" if record.circular else "linear"
    # The locus name is whitespace-delimited, so spaces become field
    # separators and a reader takes the first word as the name.  Long
    # names are kept whole rather than cut at a fixed width: both SnapGene
    # and Benchling read them, and `SOD1_in_pET-28a(` -- a name truncated
    # mid-token -- names nothing.
    locus = "_".join(record.name.split()) or "unnamed"
    out = [
        f"LOCUS       {locus:<24} {n:>7} bp    DNA     "
        f"{topology} SYN",
        f"DEFINITION  {definition or record.name}.",
        "FEATURES             Location/Qualifiers",
    ]
    for f in record.features:
        span = f"{f.start + 1}..{f.end}"
        if not f.forward:
            span = f"complement({span})"
        out.append(f"     {f.kind[:15]:<16}{span}")
        out.append(f"                     /label=\"{f.name}\"")
    out.append("ORIGIN")
    seq = record.seq.lower()
    for i in range(0, n, 60):
        chunk = seq[i : i + 60]
        groups = " ".join(chunk[j : j + 10] for j in range(0, len(chunk), 10))
        out.append(f"{i + 1:>9} {groups}")
    out.append("//")
    return "\n".join(out) + "\n"


def load(path: str | Path) -> Record:
    """Read one molecule from a .dna, .gb, .fa/.fasta, .seq or .txt file."""
    path = Path(path)
    if not path.exists():
        raise SequenceError(f"{path}: no such file")
    if path.suffix.lower() == ".dna":
        return parse_snapgene(path.read_bytes(), source=str(path))
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.lstrip().startswith(">"):
        return parse_fasta(text, source=str(path))[0]
    if text.lstrip().startswith("LOCUS"):
        return parse_genbank(text, source=str(path))
    return Record(path.stem, _clean(text, source=str(path)))


def load_sequence(value: str | Path) -> Record:
    """Read a molecule from a file path, or accept a bare sequence string."""
    text = str(value)
    candidate = re.sub(r"\s+", "", text).upper()
    if candidate and set(candidate) <= set(IUPAC) and len(candidate) >= 20:
        return Record("<literal>", candidate)
    return load(text)
