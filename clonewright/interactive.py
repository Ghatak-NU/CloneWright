"""The session that asks for what it needs, one question at a time.

`clonewright design gene.fa --vector pET-28a` assumes you already know
which vector you want and what the flags are called.  Running
`clonewright` with no arguments assumes neither: it asks for the gene,
offers the vectors it has, asks where the gene came from, and prints the
report.  Four questions, each with a default, and every answer checked
before the next one is asked.

**Input and output are injected, not called.** Everything here goes
through a `Console`, so the whole session can be driven from a list of
strings in a test -- which is the only way an interactive program gets
tested at all. `Console.standard()` wires it to the terminal.

Two rules the prompts follow, because they are what make a wizard bearable:

* **A bad answer is re-asked, not fatal.** A mistyped path costs you one
  line, not the session. This is not in tension with the refusal rule the
  pipeline enforces -- that one is about not re-asking a *tool* after it
  declines, and a person correcting a typo is not that.
* **Enter is always a valid answer.** Every question states its default,
  and the defaults alone produce a sound design.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import __version__
from . import construct as construct_mod
from . import design as design_mod
from . import ncbi
from . import orchestrate
from . import seqio
from . import source as source_mod
from . import vectors as vectors_mod
from .seqio import SequenceError
from .trace import InvariantError


class Cancelled(Exception):
    """The person stopped the session: Ctrl-C, or end of input."""


@dataclass
class Console:
    """Where questions go and answers come from."""

    read: Callable[[str], str]
    write: Callable[[str], None]

    @classmethod
    def standard(cls) -> "Console":
        def read(prompt: str) -> str:
            try:
                return input(prompt)
            except (EOFError, KeyboardInterrupt) as exc:
                raise Cancelled from exc

        def write(text: str = "") -> None:
            print(text)

        return cls(read=read, write=write)

    @classmethod
    def scripted(cls, answers: list[str]) -> tuple["Console", list[str]]:
        """A console that answers from a list and records what was printed."""
        pending = list(answers)
        printed: list[str] = []

        def read(prompt: str) -> str:
            printed.append(prompt)
            if not pending:
                raise Cancelled("the script ran out of answers")
            return pending.pop(0)

        def write(text: str = "") -> None:
            printed.append(text)

        return cls(read=read, write=write), printed

    # -- the question shapes ---------------------------------------------

    def ask(self, question: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        answer = self.read(f"{question}{suffix}\n> ").strip()
        return answer or default

    def confirm(self, question: str, *, default: bool = True) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            answer = self.read(f"{question} [{hint}]\n> ").strip().lower()
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.write("  please answer y or n.")

    def choose(self, question: str, options: list[str], *, default: int) -> int:
        """A numbered menu.  Returns a 1-based index."""
        self.write(question)
        for index, text in enumerate(options, start=1):
            self.write(f"  {index}  {text}")
        while True:
            answer = self.read(
                f"[1-{len(options)}, or Enter for {default}]\n> "
            ).strip()
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer)
            self.write(f"  pick a number between 1 and {len(options)}.")


# ---------------------------------------------------------------------------
# the questions
# ---------------------------------------------------------------------------


OWN_MAP = "a vector map of my own (a file path)"
LET_IT_CHOOSE = "design into all of them and pick the best for this gene"


def fetch_from_ncbi(console: Console, answer: str) -> tuple[seqio.Record, str]:
    """Fetch a link, and settle which coding sequence was meant.

    The choice is made here rather than left to the pipeline, because when
    a region holds two genes the person looking at the page knows which one
    they linked and the longest-wins rule is a guess. The pipeline then
    re-resolves from the cache with the answer, so the fetch happens once.
    """
    console.write("  fetching from NCBI...")
    target = ncbi.parse_target(answer)
    text, from_cache = ncbi.fetch(target)
    fetched = ncbi.parse_record(text, from_cache=from_cache)
    console.write(
        f"  {target.describe()}: {fetched.definition or fetched.accession}"
        f", {len(fetched)} bp"
        + (" (from the cache)" if from_cache else "")
    )

    usable = [c for c in fetched.coding if not c.pseudo]
    gene = ""
    if len(usable) > 1:
        picked = console.choose(
            f"  That region holds {len(usable)} coding sequences. Which one?",
            [c.label() + (f" -- {c.product}" if c.product else "")
             for c in usable],
            default=1 + max(
                range(len(usable)), key=lambda i: usable[i].length
            ),
        )
        gene = usable[picked - 1].gene or usable[picked - 1].protein_id

    resolved = ncbi.resolve(answer, gene=gene)
    for issue in resolved.issues:
        if issue.level != "note" or "splice is confirmed" in issue.text:
            console.write(f"  [{issue.level}] {issue.text}")
    if resolved.record is None:
        raise ncbi.FetchError(
            "; ".join(i.text for i in resolved.errors)
            or "nothing usable came back"
        )
    return resolved.record, gene


def ask_for_gene(console: Console) -> tuple[str, seqio.Record, str]:
    """The gene, as something the pipeline can take, and what was read.

    Returns the answer as given, the parsed record, and which coding
    sequence was chosen if the answer was an NCBI link: the pipeline wants
    the first and the third, and the person wants to be told what was
    understood before four more questions go by.
    """
    hint = (
        "a file path (FASTA, GenBank, SnapGene .dna), an NCBI link or\n"
        "accession, or paste the sequence"
    )
    if Path("examples").is_dir():
        hint += "\n(there are sequences in examples/ to try)"
    while True:
        answer = console.ask(f"Which gene are you cloning?\n{hint}")
        if not answer:
            console.write("  nothing to clone yet -- a path, a link or a "
                          "sequence.")
            continue
        cds_gene = ""
        if ncbi.looks_like_a_target(answer):
            try:
                record, cds_gene = fetch_from_ncbi(console, answer)
            except ncbi.FetchError as exc:
                console.write(f"  {exc}")
                continue
        else:
            try:
                record = seqio.load_sequence(answer)
            except SequenceError as exc:
                console.write(f"  {exc}")
                continue

        orf = design_mod.read_orf(record.seq)
        console.write(
            f"  read {record.name}, {len(record)} bp, "
            f"{record.seq[:3]}...{record.seq[-3:]}"
        )
        if orf.is_clean:
            console.write(
                "  it reads cleanly from a start codon to a stop, in frame."
            )
        else:
            console.write(
                "  note: it does not read as a clean coding sequence -- "
                + "; ".join(orf.faults())
            )
            console.write(
                "  that is often an intron or a UTR, and the design will "
                "say so rather than ignore it."
            )
        console.write()
        return answer, record, cds_gene


def ask_for_vector(console: Console) -> str:
    """A catalog name, a path, or empty to mean "choose one for me"."""
    names = list(vectors_mod.available())
    rows = []
    for name in names:
        spec = vectors_mod.CATALOG[name]
        rows.append(
            f"{spec.name:<12} {spec.promoter:<7} {spec.antibiotic:<12}"
            f"{spec.summary or spec.n_terminal}"
        )
    rows.append(LET_IT_CHOOSE)
    rows.append(OWN_MAP)
    default = len(rows) - 1  # "choose for me", which is the useful default

    picked = console.choose(
        "Which vector?", rows, default=default
    )
    console.write()
    if picked == default:
        return ""
    if picked == len(rows):
        while True:
            path = console.ask("Path to the vector map")
            if not path:
                console.write("  a path, or Ctrl-C to start over.")
                continue
            try:
                vectors_mod.resolve(path)
            except vectors_mod.VectorError as exc:
                console.write(f"  {exc}")
                continue
            console.write()
            return path
    return names[picked - 1]


def ask_for_origin(console: Console) -> str:
    """The organism, which decides the intron and rare-codon checks."""
    while True:
        answer = console.ask(
            "Where did the gene come from?\n"
            "This decides the intron and rare-codon checks, and nothing "
            "else uses it.\n"
            "A species (human, mouse, candida, plasmodium, leishmania, "
            "e. coli) or a\nclade (mammalian, yeast, bacterial, viral, "
            "synthetic). Enter to skip",
        )
        if not answer:
            console.write()
            return ""
        try:
            source_mod.get(answer)
        except KeyError as exc:
            console.write(f"  {exc}")
            continue
        console.write()
        return answer


def build_request(console: Console) -> orchestrate.Request:
    """The whole interview, as a `Request` the pipeline can run."""
    console.write(f"clonewright {__version__}")
    console.write("restriction-cloning primers for expression in E. coli")
    console.write()

    insert, record, cds_gene = ask_for_gene(console)
    insert_name = ""
    if record.name == "<literal>":
        # A pasted sequence brought no name, and the name goes on to the
        # primers and into the report's ordering table.
        insert_name = console.ask(
            "What should I call it? This names the primers", default="insert"
        )
        console.write()
    vector = ask_for_vector(console)
    origin = ask_for_origin(console)
    verify = console.confirm(
        "Also design the primers that verify the clone -- a colony screen "
        "and a\nSanger panel covering both junctions?",
        default=True,
    )
    console.write()
    return orchestrate.Request(
        insert=insert, insert_name=insert_name, cds_gene=cds_gene,
        vector=vector, origin=origin, verify=verify,
    )


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------


def offer_the_construct(console: Console, outcome: orchestrate.Outcome) -> None:
    """Offer to write the finished plasmid where they can open it."""
    if outcome.design is None or outcome.refused:
        return
    if not console.confirm(
        "Save the finished plasmid as GenBank, to open in SnapGene or "
        "Benchling?",
        default=False,
    ):
        return
    design = outcome.design
    vector_name = getattr(outcome.spec, "name", "vector")
    suggested = f"{design.insert.name}_in_{vector_name}.gb".replace(" ", "_")
    path = console.ask("Where to", default=suggested)
    record = construct_mod.assemble(
        design, name=f"{design.insert.name}_in_{vector_name}"
    )
    try:
        Path(path).write_text(
            seqio.write_genbank(
                record,
                definition=(
                    f"{design.insert.name} cloned into {vector_name} with "
                    f"{design.pair.label} by clonewright {__version__}"
                ),
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        console.write(f"  could not write it: {exc}")
        return
    console.write(
        f"  written: {path} "
        f"({len(record.features)} features carried across the ligation)"
    )


def run(console: Console | None = None) -> int:
    """One interactive session.  Returns the process exit code.

    The exit codes are the ones the rest of the command line uses -- 0 for
    a design, 1 for a refusal, 2 for input that could not be used -- so a
    session driven from a here-document scripts the same way the flags do.
    """
    console = console or Console.standard()
    try:
        request = build_request(console)
    except Cancelled:
        console.write("\nstopped.")
        return 130

    try:
        outcome = orchestrate.run(request)
    except (SequenceError, vectors_mod.VectorError,
            ncbi.FetchError) as exc:
        console.write(f"clonewright: {exc}")
        return 2
    except InvariantError as exc:
        console.write(
            f"clonewright: the answer failed its own audit ({exc})"
        )
        return 2

    console.write(outcome.answer)
    console.write()

    try:
        offer_the_construct(console, outcome)
    except Cancelled:
        pass

    if outcome.refused:
        console.write(
            "\nThat is a refusal, not a crash: the reasons are above, and "
            "the next\nstep is a different vector or a different strategy."
        )
        return 1
    return 0


def available() -> bool:
    """Whether there is a person on the other end to ask.

    Used to decide what bare `clonewright` means.  With no terminal --
    a pipe, a cron job, a CI step -- an interview would block or read a
    stray line of input as a file path, so the bare command prints its
    help there instead.
    """
    return bool(getattr(sys.stdin, "isatty", lambda: False)())
