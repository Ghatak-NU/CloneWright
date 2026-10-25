"""Fetching a gene from NCBI, and cutting the coding sequence out of it.

Paste the URL of the record you are looking at:

    https://www.ncbi.nlm.nih.gov/nuccore/NC_000003.12?report=genbank&from=129974720&to=129977935

and this module fetches that exact region, finds the coding sequences
annotated in it, splices the exons together and hands back something the
designer can use.

**The splicing is the whole job, and it is the part that can be wrong
silently.** That region of chromosome 3 carries the human TRH gene, whose
CDS is annotated `join(1098..1308,1980..2497)`: two exons with 671 bases of
intron between them. Taking the outer bounds instead -- which is what a
naive read of the location gives you -- yields a sequence that is 1400 bp
instead of 729, is not a multiple of three, and encodes nothing. It would
still amplify, still ligate, and still give colonies.

So the splice is **checked against NCBI's own answer**. A CDS feature
carries a `/translation` qualifier: the protein NCBI says that coding
sequence makes. This module splices the DNA itself, translates it, and
compares. If the two agree residue for residue, the exon boundaries, the
strand and the reading frame were all handled correctly, and it says so.
If they disagree, nothing is returned -- a primer pair built on a wrong
splice is worse than no primer pair, because it works.

Two other things it refuses rather than papers over:

* **A CDS the linked region cuts in half.** GenBank marks that with `<` or
  `>`. If the `from`/`to` in your URL does not cover the whole gene, you
  get told which end is missing and by roughly how much, rather than
  primers for a fragment.
* **A region with no coding sequence at all.** A promoter region, an
  intron, an lncRNA: there is nothing to clone for expression, and saying
  so beats designing against genomic DNA.

Every fetch is cached as the GenBank text it came from, so a design
repeated months later gets the same sequence and a revised database does
not silently change a construct. The cache is plain text, readable, and
the path is printed when it is written.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .design import Issue
from .seqio import Location, Record, parse_location, translate

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI asks that automated requests identify themselves, and that an email
# address be supplied so they can get in touch before blocking anything.
# `tool` is fixed; the address comes from the environment or the caller.
TOOL = "clonewright"
EMAIL_ENV = "CLONEWRIGHT_EMAIL"
CACHE_ENV = "CLONEWRIGHT_CACHE"

# `gbwithparts` rather than `gb`: a chromosome record is a CON record whose
# plain GenBank form carries the feature table and *no sequence*, which
# parses perfectly and designs nothing.
RETTYPE = "gbwithparts"

ACCESSION = re.compile(r"^[A-Z]{1,2}_?[0-9]{5,9}(?:\.[0-9]+)?$", re.I)


class FetchError(RuntimeError):
    """Raised when a record cannot be fetched, found or used."""


# ---------------------------------------------------------------------------
# what the user pasted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A record, and the slice of it that was asked for."""

    accession: str
    start: int | None = None   # 1-based inclusive, as NCBI writes it
    end: int | None = None
    strand: int = 1

    @property
    def is_region(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def slug(self) -> str:
        parts = [self.accession.replace("/", "_")]
        if self.is_region:
            parts.append(f"{self.start}-{self.end}")
        if self.strand == 2:
            parts.append("minus")
        return "_".join(parts)

    def describe(self) -> str:
        where = (
            f" region {self.start}-{self.end}" if self.is_region else ""
        )
        strand = " (minus strand)" if self.strand == 2 else ""
        return f"{self.accession}{where}{strand}"


def looks_like_a_target(value: str) -> bool:
    """Whether this string is something to fetch rather than open."""
    text = value.strip()
    if text.lower().startswith(("http://", "https://", "www.ncbi")):
        return True
    return bool(ACCESSION.match(text))


def parse_target(value: str) -> Target:
    """An NCBI URL or a bare accession, as something `fetch` can use.

    The URL forms accepted are the ones the nucleotide viewer produces --
    `/nuccore/<accession>`, with `report=`, `from=` and `to=` as query
    parameters, and `strand=` when you have flipped the view. A bare
    accession works too.
    """
    text = value.strip().strip('"\'')
    if ACCESSION.match(text):
        return Target(accession=text.upper())

    if not text.lower().startswith(("http://", "https://")):
        text = "https://" + text
    parsed = urllib.parse.urlparse(text)
    host = parsed.netloc.lower()
    if "ncbi.nlm.nih.gov" not in host:
        raise FetchError(
            f"{value!r} is not an NCBI URL. Paste the address of the record "
            f"you are looking at on www.ncbi.nlm.nih.gov, or give the "
            f"accession on its own"
        )

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise FetchError(f"{value!r} names no record")

    database = parts[0].lower()
    if database in ("gene", "protein", "pubmed", "sra", "bioproject"):
        raise FetchError(
            f"that is a {database} page, and this fetches nucleotide "
            f"records. On a gene page, open the sequence you want under "
            f"\"NCBI Reference Sequences\" or \"Genomic regions, "
            f"transcripts, and products\" and paste that nuccore link -- it "
            f"is the one that says which transcript you mean"
        )
    if database not in ("nuccore", "nucleotide", "nucest", "nucgss"):
        raise FetchError(
            f"{database!r} is not a nucleotide database this understands; "
            f"expected a /nuccore/ link"
        )

    accession = parts[1] if len(parts) > 1 else ""
    if not accession:
        raise FetchError(f"{value!r} names a database but no record")

    query = urllib.parse.parse_qs(parsed.query)

    def number(name: str) -> int | None:
        raw = (query.get(name) or [""])[0].replace(",", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise FetchError(
                f"{name}={raw!r} in that URL is not a position"
            ) from exc

    start, end = number("from"), number("to")
    if (start is None) != (end is None):
        raise FetchError(
            "that URL gives only one end of a region; both from= and to= "
            "are needed, or neither"
        )
    if start is not None and end is not None and start > end:
        start, end = end, start

    strand_raw = (query.get("strand") or ["1"])[0].lower()
    strand = 2 if strand_raw in ("2", "true", "minus", "-1") else 1

    return Target(accession=accession.upper(), start=start, end=end,
                  strand=strand)


# ---------------------------------------------------------------------------
# the cache, and the fetch
# ---------------------------------------------------------------------------


def default_cache() -> Path:
    """Where fetched records are kept, overridable by the environment."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".clonewright" / "ncbi"


def cache_path(target: Target, cache: Path | None = None) -> Path:
    return (cache or default_cache()) / f"{target.slug}.gb"


def read_cached(target: Target, cache: Path | None = None) -> str | None:
    path = cache_path(target, cache)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if "ORIGIN" in text else None


def write_cached(target: Target, text: str, cache: Path | None = None) -> Path:
    path = cache_path(target, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def efetch_url(target: Target, *, email: str = "") -> str:
    params = {
        "db": "nuccore",
        "id": target.accession,
        "rettype": RETTYPE,
        "retmode": "text",
        "tool": TOOL,
    }
    if target.is_region:
        params["seq_start"] = str(target.start)
        params["seq_stop"] = str(target.end)
    if target.strand == 2:
        params["strand"] = "2"
    address = email or os.environ.get(EMAIL_ENV, "")
    if address:
        params["email"] = address
    return f"{EUTILS}/efetch.fcgi?" + urllib.parse.urlencode(params)


def fetch(
    target: Target,
    *,
    allow_network: bool = True,
    cache: Path | None = None,
    timeout: float = 30.0,
    email: str = "",
) -> tuple[str, bool]:
    """The GenBank text for a target, and whether it came from the cache."""
    cached = read_cached(target, cache)
    if cached is not None:
        return cached, True
    if not allow_network:
        raise FetchError(
            f"{target.describe()} is not in the cache at "
            f"{cache_path(target, cache)}, and the network is switched off"
        )
    url = efetch_url(target, email=email)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raise FetchError(
            f"NCBI returned {exc.code} for {target.describe()}. Check the "
            f"accession, and that the region is inside the record"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError(
            f"could not reach NCBI ({exc}). The record can be downloaded by "
            f"hand from {url} and passed as a file instead"
        ) from exc

    if "ORIGIN" not in text:
        head = " ".join(text.split())[:200]
        raise FetchError(
            f"NCBI returned no sequence for {target.describe()}: {head}"
        )
    write_cached(target, text, cache)
    return text, False


# ---------------------------------------------------------------------------
# reading the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodingSequence:
    """One CDS feature, with everything needed to splice and check it."""

    location: Location
    gene: str = ""
    product: str = ""
    protein_id: str = ""
    # The protein NCBI says this CDS makes. The cross-check depends on it.
    translation: str = ""
    codon_start: int = 1
    pseudo: bool = False

    @property
    def exons(self) -> int:
        return len(self.location.segments)

    @property
    def length(self) -> int:
        return self.location.spliced_length

    def label(self) -> str:
        name = self.gene or self.product or self.protein_id or "unnamed CDS"
        exons = "1 exon" if self.exons == 1 else f"{self.exons} exons"
        strand = "+" if self.location.forward else "-"
        return f"{name} ({self.length} bp, {exons}, {strand})"


@dataclass
class FetchedRecord:
    """A fetched record: its sequence, and the CDSs annotated in it."""

    accession: str
    definition: str
    organism: str
    seq: str
    coding: list[CodingSequence] = field(default_factory=list)
    from_cache: bool = False

    def __len__(self) -> int:
        return len(self.seq)


_QUALIFIER = re.compile(r'^/([A-Za-z_]+)(?:=(.*))?$')


def _feature_blocks(text: str) -> list[tuple[str, str, dict[str, str]]]:
    """Every feature in the table, as (key, location, qualifiers).

    Written out rather than reusing `seqio.parse_genbank` because that one
    keeps a feature's outer bounds and the one qualifier it needs for a
    name. Splicing needs the location string intact and the `/translation`
    whole, and a translation is wrapped across as many lines as it takes.
    """
    if "FEATURES" not in text:
        return []
    block = text.split("FEATURES", 1)[1].split("\nORIGIN", 1)[0]
    features: list[tuple[str, str, dict[str, str]]] = []
    key = location = ""
    qualifiers: dict[str, str] = {}
    current = ""

    def flush() -> None:
        nonlocal key, location, qualifiers, current
        if key:
            features.append((key, location, qualifiers))
        key = location = current = ""
        qualifiers = {}

    for line in block.splitlines():
        if not line.strip():
            continue
        # A feature key sits at column 5, its qualifiers at column 21.
        if len(line) > 5 and line[5] != " " and not line.strip().startswith("/"):
            flush()
            parts = line.strip().split(maxsplit=1)
            key = parts[0]
            location = parts[1] if len(parts) > 1 else ""
            continue
        if not key:
            continue
        stripped = line.strip()
        match = _QUALIFIER.match(stripped)
        if match:
            current = match.group(1)
            value = (match.group(2) or "").strip()
            qualifiers[current] = value.strip('"')
            continue
        if current:
            # A wrapped qualifier value. A translation is joined with no
            # separator -- it is one word broken across lines -- while
            # prose qualifiers get a space.
            joiner = "" if current == "translation" else " "
            qualifiers[current] = (
                qualifiers[current].rstrip('"') + joiner + stripped.strip('"')
            )
        elif not location.endswith(")") or stripped.startswith(","):
            # A location wrapped onto the next line.
            location += stripped
    flush()
    return features


def parse_record(text: str, *, from_cache: bool = False) -> FetchedRecord:
    """Read fetched GenBank text into a record plus its coding sequences."""
    # Both are recorded and VERSION wins. A region record writes
    # `ACCESSION NC_000003 REGION: 129974720..129977935` and
    # `VERSION NC_000003.12`, and it is the versioned form that identifies
    # which build of the sequence this is -- the whole point of caching a
    # fetch is that a revised database cannot silently change a construct,
    # and `NC_000003` without its version does not pin that down.
    version = accession = definition = organism = ""
    for line in text.splitlines():
        if line.startswith("VERSION") and not version:
            version = line.split(maxsplit=1)[1].split()[0]
        elif line.startswith("ACCESSION") and not accession:
            accession = line.split(maxsplit=1)[1].split()[0]
        elif line.startswith("DEFINITION") and not definition:
            definition = line.split(maxsplit=1)[1].strip().rstrip(".")
        elif line.strip().startswith("ORGANISM") and not organism:
            organism = line.split("ORGANISM", 1)[1].strip()

    if "ORIGIN" not in text:
        raise FetchError("that record carries no sequence")
    body = text.split("\nORIGIN", 1)[1].split("//", 1)[0]
    seq = re.sub(r"[^A-Za-z]", "", body).upper()
    if not seq:
        raise FetchError("that record's sequence is empty")

    coding: list[CodingSequence] = []
    for key, raw_location, qualifiers in _feature_blocks(text):
        if key != "CDS":
            continue
        location = parse_location(raw_location)
        if location is None:
            continue
        try:
            codon_start = int(qualifiers.get("codon_start", "1"))
        except ValueError:
            codon_start = 1
        coding.append(CodingSequence(
            location=location,
            gene=qualifiers.get("gene", ""),
            product=qualifiers.get("product", ""),
            protein_id=qualifiers.get("protein_id", ""),
            translation=re.sub(r"\s+", "", qualifiers.get("translation", "")),
            codon_start=codon_start,
            pseudo="pseudo" in qualifiers,
        ))

    return FetchedRecord(
        accession=version or accession or "unknown",
        definition=definition,
        organism=organism,
        seq=seq,
        coding=coding,
        from_cache=from_cache,
    )


# ---------------------------------------------------------------------------
# splicing, and proving the splice
# ---------------------------------------------------------------------------


def spliced(cds: CodingSequence, seq: str) -> str:
    """The coding sequence itself: exons joined, strand and frame applied."""
    coding = cds.location.extract(seq)
    if cds.codon_start > 1:
        coding = coding[cds.codon_start - 1 :]
    return coding


def check(cds: CodingSequence, seq: str) -> list[Issue]:
    """Whether the spliced sequence is a coding sequence, and the right one.

    The last check is the one that matters. Splicing exons is arithmetic
    with several chances to be off by a base -- the exon order, the strand,
    the `codon_start` offset -- and every one of those produces a sequence
    that looks like DNA and encodes nonsense. NCBI publishes the protein it
    says this CDS makes, so the arithmetic can simply be checked against
    it, and that is a proof rather than a plausibility argument.
    """
    issues: list[Issue] = []
    coding = spliced(cds, seq)

    if cds.pseudo:
        issues.append(Issue(
            "error",
            f"{cds.label()} is annotated as a pseudogene, so its coding "
            f"sequence is not expected to make a protein",
        ))

    if cds.location.partial:
        issues.append(Issue(
            "error",
            f"the region you linked cuts {cds.gene or 'this CDS'} short: "
            f"GenBank marks it partial, so the {len(coding)} bp here is a "
            f"fragment of the coding sequence rather than the whole of it. "
            f"Widen the from= and to= in the URL, or link the mRNA record "
            f"(an NM_ accession), which carries the complete CDS",
        ))

    if len(coding) % 3:
        issues.append(Issue(
            "error",
            f"the spliced coding sequence is {len(coding)} bp, which is not "
            f"a multiple of three",
        ))

    if cds.exons > 1:
        introns = cds.location.end - cds.location.start - cds.length
        issues.append(Issue(
            "note",
            f"spliced {cds.exons} exons into {len(coding)} bp, leaving out "
            f"{introns} bp of intron. E. coli cannot splice, so this is the "
            f"sequence to clone -- and it is not the sequence you get by "
            f"amplifying the region from genomic DNA. The template has to "
            f"be cDNA or a synthetic construct",
        ))

    if cds.translation and cds.location.partial:
        # A fragment is not expected to match the whole protein, and the
        # reason is already reported above. Saying "the exon boundaries
        # have been read wrongly" here would accuse the arithmetic of a
        # fault that belongs to the requested region.
        mine = translate(coding, to_stop=True)
        issues.append(Issue(
            "note",
            f"the fragment translates to {len(mine)} residues of the "
            f"{len(cds.translation)} NCBI annotates, which is consistent "
            f"with it being cut short rather than mis-spliced",
        ))
    elif cds.translation:
        mine = translate(coding, to_stop=True)
        theirs = cds.translation
        if mine == theirs:
            issues.append(Issue(
                "note",
                f"the splice is confirmed: translating it gives the "
                f"{len(theirs)} residue protein NCBI annotates for "
                f"{cds.protein_id or cds.gene or 'this CDS'}, residue for "
                f"residue",
            ))
        else:
            where = next(
                (i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b),
                min(len(mine), len(theirs)),
            )
            issues.append(Issue(
                "error",
                f"the spliced sequence does not translate to the protein "
                f"NCBI annotates: {len(mine)} residues against "
                f"{len(theirs)}, first differing at residue {where + 1}. "
                f"The exon boundaries or the strand have been read wrongly, "
                f"and primers from this would be primers for the wrong "
                f"protein",
            ))
    else:
        issues.append(Issue(
            "warning",
            "that CDS carries no /translation, so the splice could not be "
            "checked against NCBI's own answer. The frame and stop-codon "
            "checks below are all that stand behind it",
        ))

    if not coding.startswith("ATG"):
        issues.append(Issue(
            "warning",
            f"the coding sequence starts {coding[:3]}, not ATG",
        ))
    tail = coding[-3:]
    if tail not in ("TAA", "TAG", "TGA"):
        issues.append(Issue(
            "warning",
            f"the coding sequence ends {tail}, which is not a stop codon",
        ))
    return issues


def as_record(cds: CodingSequence, fetched: FetchedRecord) -> Record:
    """The spliced CDS as a `Record` the designer can take."""
    name = cds.gene or cds.protein_id or fetched.accession
    return Record(name=name, seq=spliced(cds, fetched.seq), circular=False)


@dataclass
class Resolved:
    """What a fetch produced: the gene, and everything found on the way."""

    target: Target
    fetched: FetchedRecord
    chosen: CodingSequence | None
    record: Record | None
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return self.record is not None and not self.errors


def choose(
    fetched: FetchedRecord, *, gene: str = ""
) -> tuple[CodingSequence | None, list[Issue]]:
    """Which coding sequence in the record is the one meant.

    One CDS and there is nothing to decide. Several, and a name settles it
    if one was given; otherwise the longest is taken and the others are
    named in a note, because a region holding two genes is a region where
    the choice should be visible rather than silent.
    """
    issues: list[Issue] = []
    usable = [c for c in fetched.coding if not c.pseudo]
    if not usable:
        if fetched.coding:
            return None, [Issue(
                "error",
                "the only coding sequence in that region is annotated as a "
                "pseudogene. There is no protein to express from it",
            )]
        return None, [Issue(
            "error",
            "that region has no annotated coding sequence in it. For "
            "expression there has to be a CDS to clone -- link a region "
            "that contains the gene, or an mRNA (NM_) record, rather than "
            "a promoter, an intron or a non-coding transcript",
        )]

    if gene:
        wanted = gene.strip().lower()
        matches = [
            c for c in usable
            if wanted in (c.gene.lower(), c.protein_id.lower())
            or wanted in c.product.lower()
        ]
        if not matches:
            names = ", ".join(sorted({c.gene or c.protein_id for c in usable}))
            return None, [Issue(
                "error",
                f"no coding sequence in that region matches {gene!r}; it "
                f"holds {names}",
            )]
        if len(matches) > 1:
            issues.append(Issue(
                "note",
                f"{len(matches)} coding sequences match {gene!r}; taking the "
                f"longest",
            ))
        return max(matches, key=lambda c: c.length), issues

    if len(usable) > 1:
        others = ", ".join(c.label() for c in usable)
        issues.append(Issue(
            "warning",
            f"that region holds {len(usable)} coding sequences ({others}). "
            f"Taking the longest; name the one you want if that is not it",
        ))
    return max(usable, key=lambda c: c.length), issues


def resolve(
    value: str,
    *,
    gene: str = "",
    allow_network: bool = True,
    cache: Path | None = None,
    timeout: float = 30.0,
    email: str = "",
) -> Resolved:
    """A URL or accession to a spliced coding sequence, with the reasoning.

    Refuses by returning a `Resolved` carrying error-level issues, which
    is the shape the rest of the package refuses in. `FetchError` is kept
    for the things that stop it getting as far as a record at all: a URL
    that names nothing, a database that is not nucleotide, a network that
    cannot be reached.
    """
    target = parse_target(value)
    text, from_cache = fetch(
        target, allow_network=allow_network, cache=cache, timeout=timeout,
        email=email,
    )
    fetched = parse_record(text, from_cache=from_cache)

    issues: list[Issue] = [Issue(
        "note",
        f"fetched {target.describe()} from NCBI"
        + (" (from the local cache)" if from_cache else "")
        + f": {fetched.definition or fetched.accession}, {len(fetched)} bp",
    )]

    chosen, picked_issues = choose(fetched, gene=gene)
    issues.extend(picked_issues)
    if chosen is None:
        return Resolved(target, fetched, None, None, issues)

    issues.extend(check(chosen, fetched.seq))
    record = as_record(chosen, fetched)
    resolved = Resolved(target, fetched, chosen, record, issues)
    if resolved.errors:
        # A sequence that failed its own checks is not handed on. The
        # issues say which check and why.
        resolved.record = None
    return resolved
