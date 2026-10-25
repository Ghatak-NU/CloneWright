# clonewright

Designs restriction-cloning primers for expressing a gene in *E. coli*, and
checks them against both molecules instead of assuming they work.

Give it a coding sequence. It reads the vector, picks an enzyme pair that
cuts the backbone once each and never cuts your gene, matches the two
annealing temperatures to each other, carries the reading frame through to
the finished construct, translates the protein the construct actually
encodes, and designs the primers that prove the clone is real.

Pure Python 3, standard library only. No conda environment, no BLAST
install, no network.

```bash
python -m clonewright design examples/SOD1.fasta --vector pET-28a
```

---

## The problem it is built around

A cloning primer set is easy to produce and hard to be sure of. The
mistakes that cost a month are not arithmetic slips — they are questions
nobody asked:

- The melting temperature printed next to the primer is the melting
  temperature of **the whole oligo**, including the restriction site and
  the spacer. Only the annealing region binds in the first round of PCR.
  Those two numbers differ by several degrees, and the gel is blank.
- The enzyme pair cuts the vector in two places, or cuts inside the insert,
  or leaves ends that religate on themselves.
- The insert goes in one codon out of frame with the vector's 6xHis tag.
  Nothing about the sequence looks wrong, and the protein is nonsense.
- The gene goes into pET-28a and the construct goes into plain BL21. There
  is no T7 polymerase in plain BL21, so there is **no protein at all** —
  and the plate looks completely normal, because the kanamycin resistance
  works fine.
- The C-terminal 6xHis everyone is counting on sits downstream of the
  insert's own stop codon, so it is annotated on the map and absent from
  the protein. You find out at the nickel column.

Every one of those is checked here, and each is reported as a note, a
warning or a refusal.

---

## What one run gives you

```
-- strategy --------------------------------------------------------------
  cloning        restriction + ligation, directional
  mode           nfusion
  5' enzyme      NdeI CA^TATG (2 nt 5' overhang)   vector position 5128
  3' enzyme      XhoI C^TCGAG (4 nt 5' overhang)   vector position 5207
  excised        78 bp of vector
  insert         465 bp amplified from the template

-- primers ---------------------------------------------------------------
  FWD  SOD1_F
        5'-TGACTG CATATG ATGGCGACGAAGGCCGTGTG-3'
        full length   32 nt
        annealing     ATGGCGACGAAGGCCGTGTG  (20 nt)
        Tm annealing   70.8 C   <- sets the annealing temperature
        Tm full        76.7 C   (cycle 3 onward, once the tail is copied)
        GC             65.0 %  (annealing region)
        3' dG          -3.9 kcal/mol   GC clamp 3/5

  Tm difference  0.2 C
  suggested Ta   65.6 C  (lower annealing Tm minus 5)
  gradient       62 - 70 C
```

The primer is shown in its four parts — spacer, site, pad, annealing
region — because that is what you have to check before ordering. Both
melting temperatures are reported, and the report says which one sets the
annealing temperature.

Then the protein:

```
-- expression in E. coli -------------------------------------------------
  vector         pET-28a(+)
  promoter       T7/lac
  induction      0.1-1 mM IPTG at OD600 0.5-0.8
  selection      kanamycin
  purification   Ni-NTA on the N-terminal 6xHis; thrombin removes the tag
  cloning host   DH5-alpha (recA-, endA-)
  expression     Rosetta(DE3) (T7 RNAP (DE3), lon-/ompT-, rare tRNAs)
  plates need    kanamycin, chloramphenicol

  the protein this construct encodes:
    length       175 residues
    mass         18.2 kDa
    A280         0.33 for 1 mg/mL, 1 cm (e = 6000 M-1 cm-1)
    N terminus   MGSSHHHHHHSSGLVPRGSHM
                 (21 residues from the vector, ahead of your sequence)
    C terminus   nothing added
    stops at     TAA
```

`MGSSHHHHHHSSGLVPRGSHM` is pET-28a's documented N-terminal sequence. It is
not in a table anywhere in this package — it is translated out of the
finished construct, from the start codon the ribosome will actually use.
When the frame arithmetic is right, that string appears; when it is wrong,
it does not. The same check gives GST and MBP their real N-termini in
pGEX-4T-1 and pMAL-c5X.

Note the strain. The gene is human, its rare-codon load is above *E. coli*'s
own, so the suggestion moved from BL21(DE3) to Rosetta(DE3) — and because
Rosetta carries pRARE, chloramphenicol was added to the plates.

Finally, the two assays that prove the clone, with the primers for each: a
colony screen whose junction pairs cannot score positive on carried-over
template, and a Sanger panel tiled to cover the insert and both junctions
on both strands, using the facility's universal primers where they reach.

---

## Letting it choose the vector

```bash
python -m clonewright compare examples/SOD1.fasta
```

```
-- vectors, ranked by designing into each one ----------------------------
  vector        enzymes        mode      product          tag   warnings
  pET-28a(+)    NdeI/XhoI      nfusion   175 aa, 18 kDa   yes   1
  pMAL-c5X      SacI/PstI      nfusion   525 aa, 57 kDa   yes   1
  pET-21b(+)    NdeI/XhoI      native    154 aa, 16 kDa   no    1
  pGEX-4T-1     BamHI/NotI     nfusion   380 aa, 42 kDa   yes   2
```

The ranking is not a preference list. Every vector is **designed into**,
and what you see is what happened: a plasmid whose only useful enzymes cut
inside your gene is not a candidate for your gene, however well it suits
the protein. The ordering is fewest warnings, then whether the product
carries a tag you can purify by, then least vector sequence removed.

That middle term earns its place. pET-21b ties pET-28a on warnings and
removes slightly less vector, so on size alone it won — with a product
whose only 6xHis is C-terminal, behind the insert's own stop codon, and
therefore absent from the protein. Ranking a construct you cannot purify
above one you can is not frugality.

Run `design` with no `--vector` and it picks the top row and carries on.

---

## Asking in a sentence

```bash
python -m clonewright ask "clone examples/SOD1.fasta into pET-28a, it's a human gene"
```

```
understood: clone examples/SOD1.fasta into pET-28a
            gene from human
            mode auto, verification on
```

Patterns do the reading by default — no model, no API key, no network.
`--model` hands the sentence to Claude instead, for the phrasings patterns
do not cover (`pip install 'clonewright[agent]'`).

Either way the result is validated against the vector catalog, the mode
list and the strain table before any tool sees it. And a planner **cannot**
put a number into the answer, because `Request` has no numeric field for it
to put one in: a planner says what was asked for, and every number comes
from the computation layer.

---

## The design is built from the vector outwards

An enzyme pair is admissible only if it

1. cuts the vector exactly once each,
2. never cuts the insert, and
3. leaves ends that cannot religate on themselves.

Only once a pair survives all three are annealing regions chosen — and
their melting temperatures are matched **to each other**, not to a nominal
62 °C in isolation, because a pair that both miss 62 °C by the same amount
works and a pair that straddles it does not.

Melting temperatures are nearest-neighbour (SantaLucia 1998), with Mg²⁺ and
dNTPs folded into an equivalent monovalent concentration (von Ahsen 2001)
and the composition-dependent salt correction applied against the 1 M Na⁺
reference (Owczarzy 2004). Hairpins and dimers are scored by free energy,
weighted by whether the structure reaches the 3′ end where a polymerase can
extend it — and discounted where a primer pairs with itself across its own
palindromic restriction site, which every directional cloning primer does
and no choice of annealing region changes.

### The expression model

Which cloning mode the vector wants is read off the vector, not asked of
you:

| What the vector has | Mode | What happens |
|---|---|---|
| Its own start codon with coding sequence running up to the cloning site | `nfusion` | The insert fuses behind the vector's leader; its own ATG becomes an internal methionine |
| A cloning site running into a downstream partner | `cfusion` | The insert supplies the start codon and reads through; its stop codon is removed |
| Neither | `native` | The insert brings its own start codon and its own stop |

What makes a leader is **where it sits** — in frame, after the start codon,
running up to the cloning site — never what it is called. An earlier
version of this logic asked whether any feature was named like an affinity
tag, which read pET-22b (whose leader is the pelB *signal sequence*) as
having nothing to fuse to, fell back to `native`, and then refused every
gene, because an insert's own ATG lands 78–100 nt from that vector's
ribosome binding site.

pET-28a is worth a sentence of its own: its entire expression cassette runs
on the **reverse** strand of its own map. The T7 promoter sits at 369–386
reading backwards, down through the lac operator, the ribosome binding site
at 306–311, the ATG, the 6xHis at 270–287 and the thrombin site at 243–260,
to the polylinker at 158–242. Called directly on that map, the
frame-finding code answers about the wrong strand and does it quietly — it
comes back anchored on the start of the kanamycin resistance gene. So the
vector is turned round **once**, up front, and every piece of geometry
after that can assume a cassette that reads forward. Positions in the
report are in that turned-round frame, and the report says so and gives you
the conversion.

### The gate

A vector that does not express protein in a bacterium is refused **before
any primer is designed**:

```
-- not an E. coli expression vector --------------------------------------
   FAIL  this is a mammalian expression vector: CMV promoter is not
         transcribed by any E. coli RNA polymerase, so the construct would
         be built and selected perfectly well and express nothing.
```

Not a warning. Primers for cloning into a CMV vector answer a different
question than the one asked, and they would look exactly like an answer to
this one. pUC19 is refused too, for a different reason: it is a cloning
vector with no expression cassette at all.

---

## The agent

The pipeline runs through a tool registry, and the split that makes it
testable is this: **the route is code and only the judgement is a model.**

```bash
python -m clonewright tools
```

A planner decides what was asked for and a narrator writes the prose around
the result. Everything between them — which tools, in what order, what
happens when one declines — is a fixed pipeline, and all of it goes through
a session that enforces the contract whoever is driving. The default
planner is wholly deterministic, so the agent runs, and is tested, with no
model present at all.

Three invariants are enforced rather than described:

- **I1 — a model may describe a number; it may never supply one.** Every
  numeric token in a finished answer must be one the toolchain produced.
  That leaves exactly two honest ways to write: quote a renderer verbatim,
  or write prose with no digits in it.
- **I2 — severity propagates and never flattens.** Every issue any tool
  reported appears in the answer, under its own level. The quiet failure
  this catches is not a dropped warning but a downgraded one: nine issues
  become "design complete with some notes", every word of it true except
  the level.
- **I3 — a refusal is terminal.** The same question cannot be re-asked with
  the knobs moved. Asking about a *different* vector is a different
  question and is allowed, which is why what was asked and what it was
  asked with are recorded separately.

An answer that breaks I1 or I2 is not printed. It is not a flawed answer to
be flagged — it is not an answer, and the run exits non-zero.

---

## The bundled vectors

| Vector | Promoter | Marker | N terminus | C terminus |
|---|---|---|---|---|
| pET-28a(+) | T7/lac | kanamycin | 6xHis + thrombin site | 6xHis, past the insert's stop |
| pET-21b(+) | T7/lac | ampicillin | nothing | 6xHis, needs the stop removed |
| pGEX-4T-1 | tac | ampicillin | GST (~26 kDa) + thrombin | nothing |
| pMAL-c5X | tac | ampicillin | MBP (~42 kDa) + factor Xa | nothing |

The maps in `clonewright/data/vectors/` are plain-text GenBank, and their
**annotations are derived from the sequence**, not copied from a vendor's
feature table. `clonewright/cassette.py` finds the cassette by looking for
what defines it: the T7 consensus, σ70's two boxes at a constrained
spacing, a lac operator, a Shine–Dalgarno sequence positioned ahead of a
start codon, six histidines in frame, a protease site read as peptide, a
terminator's inverted repeat, and the polylinker as the stretch where
unique cutters crowd together. `tools/build_vector_maps.py` runs that over
a map you already have and writes the result.

The one thing it is told rather than shown is what an open reading frame
*is* — that the 816 bp ORF confers kanamycin resistance — and those names
are matched to ORFs by published coding length. An ORF with no match keeps
its measured length as its name, which reads oddly on purpose: it says what
was measured and does not say what it is.

**Every search can fail, and failing means no feature rather than a wrong
one.** That asymmetry is the whole design. An un-annotated region produces
a cautious design; a region annotated as something it is not produces a
confident wrong one. Two bugs during development were exactly that:

- A chance 111-codon ORF on the reverse strand of pGEX-4T-1's GST gene was
  annotated, counted as conflicting evidence about the reading frame, and
  turned a vector whose entire purpose is an N-terminal GST fusion into a
  `native` clone. The design was still sound and silently answered a
  different question.
- pET-21b's derived polylinker reached back past its own ribosome binding
  site and picked up an XbaI site from the upstream region. The designer
  was offered a 5′ enzyme that puts the start codon 29 nt *upstream* of the
  RBS, and refused it — correctly. The annotation was wrong to offer it.

Both are now tests.

### Bring your own vector

Any GenBank, SnapGene `.dna` or FASTA map works as a path:

```bash
python -m clonewright design my-gene.fasta --vector ./my-pET-28a.dna
```

A path always beats a catalog name — somebody who passes a path means the
file in front of them, quite possibly their own corrected map.

If your vector arrived as bare sequence with no annotations, derive them:

```bash
python -m clonewright annotate unannotated.fa -o annotated.gb
```

---

## Commands

| Command | What it does |
|---|---|
| `design GENE [--vector V]` | The whole pipeline. Omit `--vector` and it chooses |
| `compare GENE` | Every bundled vector, ranked by designing into each |
| `ask "sentence"` | Reads the request out of a sentence, then runs it |
| `annotate VECTOR` | Derives a bacterial cassette from sequence |
| `vectors` | The bundled catalog |
| `tools` | The tool registry the agent drives |

Useful flags on `design`: `--origin human` (turns on the intron and
rare-codon checks), `--mode`, `--expression-strain`, `--no-verify`,
`--compare`, `--json`, and `--emit-construct out.gb`, which writes the
finished plasmid as GenBank with the features that survived the ligation —
openable in SnapGene or Benchling.

Exit status is `0` for a design that succeeded, `1` for one that was
refused, and `2` for a bad invocation or an unreadable file. **A refusal is
a result, not a crash.**

---

## Install and test

```bash
git clone https://github.com/Ghatak-NU/clonewright
cd clonewright
python -m clonewright design examples/SOD1.fasta --vector pET-28a
```

Nothing to install — it runs from the checkout. For the `clonewright`
command on your PATH, `pip install -e .`.

```bash
python -m unittest discover -s tests
```

128 tests, no network, a few seconds. They cover the published
thermodynamic values, the GenBank round trip, cassette derivation on all
four bundled maps, the enzyme geometry, the frame and product arithmetic
(including that pET-28a leader), the strain checks, the three invariants,
and the CLI's exit codes. Developed on Python 3.14; the code is standard
library only and uses nothing newer than 3.10 syntax.

---

## What it does not do

Worth saying plainly, because a tool that overstates its scope wastes more
time than one that does less:

- **Restriction cloning only.** No Gibson, no SLIC, no Golden Gate, no
  TOPO. Those are different primer problems.
- ***E. coli* expression only.** A mammalian or yeast vector is refused by
  design. The engine underneath is host-aware, but this is not the tool for
  that job.
- **No codon optimisation and no gene synthesis.** Rare codons are reported
  and point at a strain; the sequence is never rewritten.
- **Four bundled vectors.** Any other vector works as a map file, but only
  these four come with their catalog facts.
- **It does not know your bench.** Enzyme buffers and incubation
  temperatures are transcribed from vendor tables, printed rather than only
  acted on, and worth checking against the tubes in your freezer.
- **A clean report is not a successful clone.** It means the checks this
  package knows how to make all passed.

---

## Provenance

The computation layer — nearest-neighbour thermodynamics, the enzyme
geometry, the frame arithmetic, the verification planner — is carried over
from BHAGWAN, a larger multi-host cloning toolkit I wrote and validated
against a set of real constructs. clonewright narrows it to one job,
*E. coli* expression, adds the expression model's front door, the
sequence-derived vector maps, the protein the construct actually encodes,
and the audited agent loop around all of it.

## License

MIT. See [LICENSE](LICENSE).
