"""Test suite.  Run with:  py -m unittest discover -s tests -v

Written against the behaviours that are load-bearing, and in particular
against the ones that were *wrong* at some point during development.  A
test whose docstring names a specific vector is usually there because that
vector broke something: pGEX-4T-1's GST fusion read as native, pET-21b
offered an XbaI site upstream of its own ribosome binding site, pET-28a's
cassette runs backwards along its map.

The published-value tests are the other kind.  They check arithmetic
against numbers from the literature rather than against this code's own
output, so a refactor that quietly changes a melting temperature fails
here instead of at a bench three weeks later.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clonewright import agent  # noqa: E402
from clonewright import cassette  # noqa: E402
from clonewright import cli  # noqa: E402
from clonewright import construct  # noqa: E402
from clonewright import design as dz  # noqa: E402
from clonewright import enzymes as ez  # noqa: E402
from clonewright import expression  # noqa: E402
from clonewright import interactive  # noqa: E402
from clonewright import ncbi  # noqa: E402
from clonewright import orchestrate  # noqa: E402
from clonewright import planner  # noqa: E402
from clonewright import report  # noqa: E402
from clonewright import seqio  # noqa: E402
from clonewright import source  # noqa: E402
from clonewright import strains  # noqa: E402
from clonewright import thermo  # noqa: E402
from clonewright import trace as tracing  # noqa: E402
from clonewright import vectors  # noqa: E402
from clonewright.oligo import Oligo  # noqa: E402

EXAMPLES = ROOT / "examples"
MAPS = ROOT / "clonewright" / "data" / "vectors"


def gene(name: str) -> seqio.Record:
    return seqio.load(EXAMPLES / f"{name}.fasta")


def vector(name: str) -> seqio.Record:
    return seqio.load(MAPS / f"{name}.gb")


# A plasmid that reads as bacterial and offers no unique cutter anywhere,
# which is the shortest route to a genuine refusal: `design` raises only
# when no enzyme pair brackets the target region.
FLAT_VECTOR = seqio.Record(
    "flat", "ATAAATTTAAATTTAAA" * 30, True,
    [
        seqio.Feature("T7 promoter", "promoter", 0, 18, True),
        seqio.Feature("RBS", "RBS", 20, 26, True),
    ],
)


# ---------------------------------------------------------------------------
# sequence input and output
# ---------------------------------------------------------------------------


class TestSeqIO(unittest.TestCase):
    def test_revcomp_is_an_involution(self):
        seq = "ATGGCGACGAAGGCCGTGTGCGTGCTGAAG"
        self.assertEqual(seqio.revcomp(seqio.revcomp(seq)), seq)

    def test_revcomp_handles_ambiguity_codes(self):
        self.assertEqual(seqio.revcomp("ACGTRYN"), "NRYACGT")

    def test_translate_matches_the_standard_code(self):
        # One codon per amino acid, read off a printed table.
        self.assertEqual(seqio.translate("ATGGCGTGGTAA"), "MAW*")
        self.assertEqual(seqio.translate("ATGGCGTGGTAA", to_stop=True), "MAW")

    def test_translate_calls_an_ambiguous_codon_X(self):
        self.assertEqual(seqio.translate("ATGNNN"), "MX")

    def test_fasta_round_trip(self):
        record = gene("SOD1")
        self.assertEqual(record.name, "SOD1")
        self.assertEqual(len(record), 465)
        self.assertTrue(record.seq.startswith("ATG"))

    def test_non_dna_is_refused(self):
        with self.assertRaises(seqio.SequenceError):
            seqio.parse_fasta(">x\nATGZZZATG\n")

    def test_genbank_round_trip_keeps_features_and_topology(self):
        original = vector("pET-28a")
        text = seqio.write_genbank(original)
        reread = seqio.parse_genbank(text, source="roundtrip.gb")
        self.assertEqual(reread.seq, original.seq)
        self.assertTrue(reread.circular)
        self.assertEqual(
            [(f.name, f.start, f.end, f.forward) for f in reread.features],
            [(f.name, f.start, f.end, f.forward) for f in original.features],
        )

    def test_genbank_complement_locations_survive(self):
        original = vector("pET-28a")
        reverse = [f for f in original.features if not f.forward]
        self.assertTrue(reverse, "pET-28a's cassette features are reverse")
        text = seqio.write_genbank(original)
        reread = seqio.parse_genbank(text)
        self.assertEqual(
            sum(1 for f in reread.features if not f.forward), len(reverse)
        )

    def test_genbank_without_origin_is_refused(self):
        with self.assertRaises(seqio.SequenceError):
            seqio.parse_genbank("LOCUS  x  10 bp\nFEATURES\n")

    def test_locus_name_is_not_truncated_mid_token(self):
        # `SOD1_in_pET-28a(` named nothing; the whole name is kept.
        record = seqio.Record("SOD1_in_pET-28a(+)", "ATGC" * 10, True)
        line = seqio.write_genbank(record).splitlines()[0]
        self.assertIn("SOD1_in_pET-28a(+)", line)

    def test_window_wraps_only_when_circular(self):
        linear = seqio.Record("l", "AAAATTTT", False)
        circular = seqio.Record("c", "AAAATTTT", True)
        self.assertEqual(linear.window(6, 2), "")
        self.assertEqual(circular.window(6, 2), "TTAA")

    def test_reverse_complement_carries_features(self):
        record = seqio.Record(
            "v", "ATGCGATCGATCG", True,
            [seqio.Feature("tag", "CDS", 0, 3, True)],
        )
        flipped = record.reverse_complement()
        f = flipped.features[0]
        self.assertEqual((f.start, f.end, f.forward), (10, 13, False))


# ---------------------------------------------------------------------------
# thermodynamics
# ---------------------------------------------------------------------------


class TestThermo(unittest.TestCase):
    def test_nearest_neighbour_table_is_symmetric(self):
        for step, value in thermo.NN_DH.items():
            mirror = seqio.revcomp(step)
            self.assertAlmostEqual(thermo.NN_DH[mirror], value, places=6)

    def test_salt_correction_never_exceeds_the_reference(self):
        # The 1 M Na+ value is the ceiling; an earlier divalent equation
        # returned temperatures above it, which cannot be right.
        for seq in ("ATGGCGACGAAGGCC", "TTTTTTTTTTAAAAA", "GCGCGCGCGCGC"):
            ref = thermo._tm_1m_na(seq, thermo.DEFAULT_CONDITIONS.primer)
            self.assertLessEqual(thermo.tm(seq), ref)

    def test_tm_rises_with_gc(self):
        self.assertGreater(thermo.tm("GCGCGCGCGCGCGCGCGCGC"),
                           thermo.tm("ATATATATATATATATATAT"))

    def test_ambiguous_bases_are_refused(self):
        with self.assertRaises(thermo.ThermoError):
            thermo.tm("ATGCN")

    def test_longest_run_and_gc_clamp(self):
        self.assertEqual(thermo.longest_run("AAGGGGTC"), 4)
        self.assertEqual(thermo.gc_clamp("AAAAAGGCCC"), 5)


class TestOligo(unittest.TestCase):
    def test_tm_of_the_annealing_region_is_not_the_tm_of_the_oligo(self):
        o = Oligo("p", spacer="TGACTG", site="CATATG",
                  anneal="ATGGCGACGAAGGCCGTGTG")
        self.assertNotAlmostEqual(o.tm_anneal, o.tm_full, places=1)
        self.assertLess(o.tm_anneal, o.tm_full)

    def test_formatted_shows_the_four_parts(self):
        o = Oligo("p", spacer="TGACTG", site="CATATG", anneal="ATGGCG")
        self.assertEqual(o.formatted(), "5'-TGACTG CATATG ATGGCG-3'")

    def test_palindromic_site_dimer_is_discounted(self):
        # Every directional cloning primer self-pairs across its own site;
        # no choice of annealing region changes it, so it must not drive
        # the ranking.
        o = Oligo("p", spacer="TGACTG", site="GAATTC",
                  anneal="ATGGCGACGAAGGCCGTGTG")
        if o.dimer_is_site_palindrome:
            self.assertLess(o.structure_penalty, -o.self_dimer.dg)


# ---------------------------------------------------------------------------
# enzymes
# ---------------------------------------------------------------------------


class TestEnzymes(unittest.TestCase):
    def test_sites_are_found_across_the_origin_only_when_circular(self):
        seq = "TCGAGxxxxxxxxxxxxxxxxxxxxC".replace("x", "A")
        self.assertEqual(ez.find_sites(seq, "CTCGAG", circular=False), [])
        self.assertEqual(ez.find_sites(seq, "CTCGAG", circular=True),
                         [len(seq) - 1])

    def test_ambiguous_recognition_sites_expand(self):
        # CCWGG: W is A or T, and both must match.
        self.assertEqual(len(ez.find_sites("CCAGGCCTGG", "CCWGG")), 2)

    def test_every_bundled_vector_is_cut_once_by_its_cloning_enzymes(self):
        for name in vectors.available():
            record = vector(name)
            unique = [
                enzyme for enzyme in ez.ENZYMES.values()
                if ez.cut_count(record.seq, enzyme, circular=True) == 1
            ]
            self.assertGreaterEqual(
                len(unique), 5, f"{name} should offer several unique cutters"
            )


# ---------------------------------------------------------------------------
# deriving the cassette from sequence
# ---------------------------------------------------------------------------


class TestCassette(unittest.TestCase):
    """`cassette` must fail by finding nothing, never by finding wrongly."""

    def test_t7_promoter_is_found_on_both_strands(self):
        seq = "A" * 50 + cassette.T7_CONSENSUS + "A" * 50
        found = cassette.t7_promoters(seq)
        self.assertEqual([f.forward for f in found], [True])
        reverse = cassette.t7_promoters(
            seqio.revcomp(seq)
        )
        self.assertEqual([f.forward for f in reverse], [False])

    def test_a_sequence_with_no_motifs_gets_no_features(self):
        bare = seqio.Record("bare", "AT" * 1500, True)
        derived = cassette.annotate(bare)
        self.assertEqual(derived.features, [])

    def test_existing_annotations_are_kept(self):
        record = seqio.Record(
            "v", "ATGC" * 500, True,
            [seqio.Feature("mine", "misc_feature", 0, 12, True)],
        )
        derived = cassette.annotate(record)
        self.assertIn("mine", [f.name for f in derived.features])

    def test_peptide_tags_are_found_by_translation_not_by_dna(self):
        # Two different codon choices for the same hexahistidine.
        for his in ("CATCATCATCATCATCAC", "CACCACCATCATCACCAC"):
            seq = "ATG" + his + "TAA"
            found = cassette.peptide_features(
                seq, table={"6xHis tag": "HHHHHH"}, kind="CDS"
            )
            self.assertTrue(
                any(f.name == "6xHis tag" for f in found), his
            )

    def test_orf_antisense_to_a_longer_one_is_dropped(self):
        """The bug: a chance ORF on the reverse strand of GST made
        pGEX-4T-1's fusion frame unconfirmed, and the vector came back as
        a native clone."""
        gst = [f for f in vector("pGEX-4T-1").features if f.name == "GST"]
        self.assertEqual(len(gst), 1)
        overlapping = [
            f for f in vector("pGEX-4T-1").features
            if f.kind == "CDS" and not f.forward
            and f.start < gst[0].end and gst[0].start < f.end
        ]
        self.assertEqual(overlapping, [])

    def test_polylinker_does_not_reach_past_the_ribosome_binding_site(self):
        """The bug: pET-21b's derived MCS included an XbaI site upstream of
        its RBS, so the designer was offered a 5' site that puts the start
        codon 29 nt before the sequence that positions the ribosome."""
        record = vector("pET-21b")
        mcs = [f for f in record.features if f.name == "MCS"]
        rbs = [f for f in record.features if f.kind == "RBS"]
        self.assertTrue(mcs and rbs)
        # The cassette runs against this map, so the polylinker must end
        # before the RBS begins.
        self.assertLessEqual(mcs[0].end, rbs[0].start)

    def test_derived_orfs_without_a_hint_are_named_by_their_length(self):
        unnamed = [
            f for f in vector("pET-28a").features
            if f.name.startswith("unnamed ORF")
        ]
        self.assertTrue(unnamed)
        for f in unnamed:
            self.assertIn("aa)", f.name)

    def test_hinted_orf_lengths_name_the_marker(self):
        names = [f.name for f in vector("pET-28a").features]
        self.assertIn("KanR", names)


# ---------------------------------------------------------------------------
# reading a vector
# ---------------------------------------------------------------------------


class TestVectorReading(unittest.TestCase):
    def test_all_bundled_vectors_read_as_bacterial(self):
        for name in vectors.available():
            self.assertEqual(dz.detect_host(vector(name)), "bacterial", name)

    def test_pet28a_cassette_runs_against_its_map(self):
        self.assertEqual(dz.expression_direction(vector("pET-28a")), -1)

    def test_inspect_flips_the_vector_so_the_frame_is_right(self):
        """Called directly on pET-28a, `fusion_frame` anchors on a
        resistance marker; `inspect` does the flip first."""
        view = dz.inspect(vector("pET-28a"))
        self.assertTrue(view.flipped)
        self.assertEqual(view.mode, "nfusion")
        self.assertIsNotNone(view.frame)
        confirmed = [f.name for f in view.frame.confirmed_by]
        self.assertIn("6xHis tag", confirmed)

    def test_fusion_vectors_choose_a_fusion_mode(self):
        for name in ("pET-28a", "pGEX-4T-1", "pMAL-c5X"):
            self.assertEqual(dz.inspect(vector(name)).mode, "nfusion", name)

    def test_a_vector_with_no_leader_chooses_native(self):
        # pET-21b's start codon is inside its NdeI site: the insert brings
        # its own ATG and there is nothing upstream to fuse behind.
        self.assertEqual(dz.inspect(vector("pET-21b")).mode, "native")

    def test_driving_promoter_is_a_promoter_feature_not_a_name_match(self):
        promoter = dz.driving_promoter(vector("pET-28a"))
        self.assertIsNotNone(promoter)
        self.assertEqual(promoter.name, "T7 promoter")
        self.assertEqual(promoter.kind, "promoter")


# ---------------------------------------------------------------------------
# the catalog
# ---------------------------------------------------------------------------


class TestCatalog(unittest.TestCase):
    def test_every_catalog_entry_has_a_map(self):
        self.assertEqual(sorted(vectors.available()),
                         sorted(vectors.CATALOG))

    def test_names_resolve_in_any_spelling(self):
        for spelling in ("pET-28a", "pet28a", "PET-28A(+)", "pET-28a(+)"):
            record, spec = vectors.resolve(spelling)
            self.assertEqual(spec.name, "pET-28a(+)", spelling)

    def test_a_path_beats_a_catalog_name(self):
        """Somebody who passes a path means the file in front of them,
        quite possibly their own corrected map."""
        with tempfile.TemporaryDirectory() as tmp:
            mine = Path(tmp) / "pET-28a.gb"
            edited = vector("pET-28a")
            edited.seq = edited.seq[:-10]  # a different molecule
            mine.write_text(seqio.write_genbank(edited), encoding="utf-8")
            record, _ = vectors.resolve(str(mine))
            self.assertEqual(len(record), len(edited))
            self.assertNotEqual(len(record), len(vector("pET-28a")))

    def test_an_unknown_name_is_refused_with_the_known_ones(self):
        with self.assertRaises(vectors.VectorError) as caught:
            vectors.resolve("pBluescript")
        self.assertIn("pET-28a", str(caught.exception))

    def test_shortlist_ranks_a_purifiable_product_above_an_untagged_one(self):
        """pET-21b removes less vector than pET-28a and ties on warnings,
        so on size alone it won -- with a product whose only tag sits past
        the insert's stop codon and is never translated."""
        usable, _ = vectors.shortlist(gene("SOD1"),
                                      opts=dz.Options(mode="auto"))
        self.assertTrue(usable)
        self.assertTrue(usable[0].tagged)
        self.assertEqual(usable[0].spec.name, "pET-28a(+)")

    def test_shortlist_rows_carry_no_issues(self):
        """A comparison row is a summary, not another result: carrying the
        design would carry its issues, and every issue in an answer's
        provenance has to be printed under its own severity."""
        usable, rejected = vectors.shortlist(gene("SOD1"))
        for row in list(usable) + list(rejected):
            self.assertEqual(tracing.issues_in(row), ())


# ---------------------------------------------------------------------------
# designing
# ---------------------------------------------------------------------------


class TestDesign(unittest.TestCase):
    def test_sod1_into_pet28a_uses_the_canonical_pair(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        self.assertTrue(d.ok)
        self.assertEqual(d.pair.label, "NdeI/XhoI")
        self.assertEqual(d.options.mode, "nfusion")

    def test_the_chosen_enzymes_do_not_cut_the_insert(self):
        for name in vectors.available():
            d = dz.design(gene("SOD1"), vector(name), dz.Options(mode="auto"))
            if not d.ok:
                continue
            for enzyme in (d.pair.five, d.pair.three):
                self.assertEqual(
                    ez.cut_count(d.insert.seq, enzyme), 0,
                    f"{enzyme.name} cuts the insert ({name})",
                )

    def test_the_chosen_enzymes_cut_the_vector_exactly_once(self):
        for name in vectors.available():
            record = vector(name)
            d = dz.design(gene("SOD1"), record, dz.Options(mode="auto"))
            if not d.ok:
                continue
            for enzyme in (d.pair.five, d.pair.three):
                self.assertEqual(
                    ez.cut_count(record.seq, enzyme, circular=True), 1,
                    f"{enzyme.name} is not a unique cutter in {name}",
                )

    def test_annealing_temperatures_are_matched_to_each_other(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        gap = abs(d.forward.tm_anneal - d.reverse.tm_anneal)
        self.assertLessEqual(gap, d.options.max_tm_mismatch)

    def test_the_construct_is_the_backbone_plus_the_digested_insert(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        self.assertEqual(len(d.construct),
                         len(d.backbone) + len(d.insert_fragment))

    def test_both_sites_are_regenerated_in_the_construct(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        for enzyme in (d.pair.five, d.pair.three):
            self.assertEqual(
                len(ez.find_sites(d.construct, enzyme.site, circular=True)), 1,
                f"{enzyme.name} should cut the product exactly once",
            )

    def test_design_is_deterministic(self):
        first = dz.design(gene("SOD1"), vector("pET-28a"),
                          dz.Options(mode="auto"))
        second = dz.design(gene("SOD1"), vector("pET-28a"),
                           dz.Options(mode="auto"))
        self.assertEqual(first.forward.sequence, second.forward.sequence)
        self.assertEqual(first.reverse.sequence, second.reverse.sequence)

    def test_every_gene_in_examples_designs_or_explains_itself(self):
        for name in ("SOD1", "GAPDH", "TNFRSF10A"):
            d = dz.design(gene(name), vector("pET-28a"),
                          dz.Options(mode="auto"))
            if not d.ok:
                self.assertTrue(d.errors, f"{name} failed with no reason")
            else:
                self.assertTrue(d.forward.anneal and d.reverse.anneal)

    def test_a_refusal_carries_its_reasons(self):
        """A vector offering no unique cutter cannot be cloned into, and
        the refusal says which check failed rather than returning primers
        that will not survive the digest."""
        with self.assertRaises(dz.DesignFailed) as caught:
            dz.design(gene("SOD1"), FLAT_VECTOR, dz.Options(mode="auto"))
        self.assertIn("single-cutter", str(caught.exception))
        self.assertTrue(getattr(caught.exception, "issues", None))

    def test_an_insert_too_short_to_prime_is_refused_not_crashed(self):
        """It used to raise a bare ValueError out of the primer picker,
        which is not an exception `design` declares, so it left the
        command line as a traceback. An NCBI region whose only coding
        sequence is a short peptide reaches this honestly."""
        tiny = seqio.Record("tiny", "ATGTGGAAATAA")
        with self.assertRaises(dz.DesignFailed) as caught:
            dz.design(tiny, vector("pET-28a"), dz.Options(mode="auto"))
        text = " ".join(i.text for i in caught.exception.issues)
        self.assertIn("12 bp", text)
        self.assertIn("synthesised oligonucleotides", text)

    def test_a_short_insert_exits_cleanly_through_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.fa"
            path.write_text(">tiny\nATGTGGAAATAA\n", encoding="utf-8")
            code, text, err = run_cli(
                "design", str(path), "--vector", "pET-28a", "--no-verify",
            )
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", text + err)

    def test_a_raised_refusal_carries_error_level_issues_only(self):
        try:
            dz.design(gene("SOD1"), FLAT_VECTOR, dz.Options(mode="auto"))
        except dz.DesignFailed as exc:
            issues = getattr(exc, "issues", ())
            self.assertTrue(issues)
            # At least one error, and the notes that led up to it come
            # along too: the reasoning is part of the refusal.
            self.assertTrue(any(i.level == "error" for i in issues))
        else:
            self.fail("the flat vector should have been refused")


# ---------------------------------------------------------------------------
# the protein
# ---------------------------------------------------------------------------


class TestProduct(unittest.TestCase):
    def test_pet28a_adds_its_published_n_terminal_sequence(self):
        """The strongest check in the suite: the leader translated out of
        the construct is pET-28a's documented N terminus, which nothing in
        this package was told."""
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        self.assertIsNotNone(made)
        self.assertEqual(made.leader, "MGSSHHHHHHSSGLVPRGSHM")
        self.assertIn("HHHHHH", made.leader)
        self.assertIn("LVPRGS", made.leader)  # the thrombin site

    def test_gst_and_mbp_fusions_begin_with_their_real_proteins(self):
        for name, opening in (("pGEX-4T-1", "MSPILGYWKIKGL"),
                              ("pMAL-c5X", "MKIEEGKLVIWINGDKGYNGLAEVGKKF")):
            d = dz.design(gene("SOD1"), vector(name), dz.Options(mode="auto"))
            made = expression.product_of(d)
            self.assertIsNotNone(made, name)
            self.assertTrue(made.leader.startswith(opening), name)

    def test_the_body_is_the_insert_translated(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        expected = seqio.translate(gene("SOD1").seq, to_stop=True)
        self.assertEqual(made.body, expected)

    def test_a_native_clone_adds_nothing(self):
        d = dz.design(gene("SOD1"), vector("pET-21b"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        self.assertEqual(made.leader, "")

    def test_a_c_terminal_tag_behind_a_stop_codon_is_not_on_the_product(self):
        """pET-21b's 6xHis is annotated on the vector and absent from the
        protein, because the insert keeps its own stop codon."""
        d = dz.design(gene("SOD1"), vector("pET-21b"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        self.assertEqual(made.trailer, "")
        self.assertNotIn("HHHHHH", made.peptide)

    def test_mass_matches_the_sum_of_its_residues(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        by_hand = sum(
            expression.RESIDUE_MASS[r] for r in made.peptide
        ) + expression.WATER_MASS
        self.assertAlmostEqual(made.mass_kda, by_hand / 1000.0, places=6)

    def test_sod1_in_pet28a_weighs_about_eighteen_kilodaltons(self):
        # SOD1 is ~16 kDa; the His-thrombin leader adds ~2.
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        made = expression.product_of(d)
        self.assertGreater(made.mass_kda, 17.0)
        self.assertLess(made.mass_kda, 19.5)

    def test_extinction_coefficient_counts_only_w_y_and_c(self):
        made = expression.Product(
            peptide="WWYC", leader="", body="WWYC", trailer="",
            start=0, stop=None,
        )
        self.assertEqual(
            made.extinction, 2 * 5500.0 + 1490.0 + 125.0
        )

    def test_a_product_with_no_aromatics_cannot_be_read_at_280(self):
        made = expression.Product(
            peptide="AAAGGG", leader="", body="AAAGGG", trailer="",
            start=0, stop=None,
        )
        self.assertEqual(made.extinction, 0.0)
        self.assertEqual(made.a280_for_1mg_ml, 0.0)


# ---------------------------------------------------------------------------
# the E. coli gate
# ---------------------------------------------------------------------------


class TestGate(unittest.TestCase):
    def test_bacterial_vectors_pass(self):
        for name in vectors.available():
            view = dz.inspect(vector(name))
            self.assertEqual(expression.suits_ecoli(view), [], name)

    def test_a_vector_with_no_bacterial_evidence_is_refused(self):
        bare = seqio.Record("mystery", "ATGC" * 1000, True)
        issues = expression.suits_ecoli(dz.inspect(bare))
        self.assertTrue(issues)
        self.assertEqual(issues[0].level, "error")
        self.assertIn("annotate", issues[0].text)

    def test_a_mammalian_vector_is_refused_as_mammalian(self):
        mammalian = seqio.Record(
            "pCMV-ish", "ATGC" * 1000, True,
            [seqio.Feature("CMV promoter", "promoter", 10, 30, True)],
        )
        issues = expression.suits_ecoli(dz.inspect(mammalian))
        self.assertTrue(issues)
        self.assertIn("mammalian", issues[0].text)


# ---------------------------------------------------------------------------
# strains
# ---------------------------------------------------------------------------


class TestStrains(unittest.TestCase):
    def test_a_t7_vector_in_plain_bl21_is_an_error(self):
        """The trap: no protein at all, and a plate that looks normal."""
        plan = strains.plan_hosts(
            dz.inspect(vector("pET-28a")).reading,
            expression=strains.get("bl21"),
        )
        self.assertFalse(plan.ok)
        self.assertTrue(
            any("no protein at all" in i.text for i in plan.errors)
        )

    def test_a_t7_vector_in_bl21de3_is_fine(self):
        plan = strains.plan_hosts(
            dz.inspect(vector("pET-28a")).reading,
            expression=strains.get("bl21de3"),
        )
        self.assertTrue(plan.ok)

    def test_a_tac_vector_needs_no_lysogen(self):
        plan = strains.plan_hosts(
            dz.inspect(vector("pGEX-4T-1")).reading,
            expression=strains.get("bl21"),
        )
        self.assertTrue(plan.ok)
        self.assertTrue(
            any("plain BL21" in i.text for i in plan.issues)
        )

    def test_a_resident_plasmid_adds_a_second_antibiotic(self):
        plan = strains.plan_hosts(
            dz.inspect(vector("pET-28a")).reading,
            expression=strains.get("rosettade3"),
            vector_antibiotic="kanamycin",
        )
        self.assertIn("chloramphenicol", plan.antibiotics)

    def test_rare_codons_point_at_rosetta(self):
        self.assertEqual(
            strains.suggest_expression_strain(rare_codons=True).name,
            "Rosetta(DE3)",
        )


# ---------------------------------------------------------------------------
# the finished plasmid
# ---------------------------------------------------------------------------


class TestConstruct(unittest.TestCase):
    def test_assembled_construct_matches_the_design(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        record = construct.assemble(d)
        self.assertEqual(record.seq, d.construct)
        self.assertTrue(record.circular)

    def test_the_insert_is_annotated_on_it(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        record = construct.assemble(d)
        self.assertIn("SOD1", [f.name for f in record.features])

    def test_no_feature_runs_off_the_end(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        record = construct.assemble(d)
        for f in record.features:
            self.assertLessEqual(f.end, len(record))
            self.assertGreaterEqual(f.start, 0)

    def test_the_six_his_leader_survives_the_ligation(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        record = construct.assemble(d)
        self.assertIn("6xHis tag", [f.name for f in record.features])

    def test_it_round_trips_through_genbank(self):
        d = dz.design(gene("SOD1"), vector("pET-28a"),
                      dz.Options(mode="auto"))
        record = construct.assemble(d)
        reread = seqio.parse_genbank(seqio.write_genbank(record))
        self.assertEqual(reread.seq, record.seq)
        self.assertEqual(len(reread.features), len(record.features))


# ---------------------------------------------------------------------------
# the agent: the registry and its gate
# ---------------------------------------------------------------------------


class TestAgent(unittest.TestCase):
    def test_every_tool_in_the_registry_is_callable(self):
        for name, tool in agent.REGISTRY.items():
            self.assertTrue(callable(tool.call), name)
            self.assertIn(tool.layer, (0, 1, 4), name)
            self.assertTrue(tool.summary, f"{name} has no summary")

    def test_a_tool_outside_the_registry_is_refused(self):
        session = agent.Session()
        with self.assertRaises(tracing.InvariantError):
            session.call("design.secret_sauce")

    def test_a_call_is_recorded_with_its_outcome(self):
        session = agent.Session()
        session.call("seqio.load", path=str(EXAMPLES / "SOD1.fasta"))
        self.assertEqual(len(session.trace.calls), 1)
        self.assertEqual(session.trace.calls[0].outcome, "returned")
        self.assertEqual(session.trace.calls[0].layer, 0)

    def test_a_rendered_block_puts_its_numbers_in_the_pool(self):
        """How I1 is satisfiable at all: a renderer is a registered tool,
        so the numbers it prints are numbers the toolchain produced."""
        session = agent.Session()
        d = session.call("design.design", insert=gene("SOD1"),
                         vector=vector("pET-28a"),
                         opts=dz.Options(mode="auto"))
        session.call("report.render", d=d)
        self.assertIn(float(len(d.amplicon)), session.trace.numbers())

    def test_i3_a_refusal_cannot_be_retried_with_the_knobs_moved(self):
        session = agent.Session()
        with self.assertRaises(dz.DesignFailed):
            session.call("design.design", insert=gene("SOD1"),
                         vector=FLAT_VECTOR, opts=dz.Options(mode="auto"))
        with self.assertRaises(tracing.InvariantError) as caught:
            session.call("design.design", insert=gene("SOD1"),
                         vector=FLAT_VECTOR,
                         opts=dz.Options(mode="auto", target_tm=55.0))
        self.assertEqual(caught.exception.invariant, "I3")

    def test_i3_allows_the_same_tool_on_a_different_target(self):
        session = agent.Session()
        with self.assertRaises(dz.DesignFailed):
            session.call("design.design", insert=gene("SOD1"),
                         vector=FLAT_VECTOR, opts=dz.Options(mode="auto"))
        # A different vector is a different question, not a retry.
        result = session.call("design.design", insert=gene("SOD1"),
                              vector=vector("pET-28a"),
                              opts=dz.Options(mode="auto"))
        self.assertTrue(result.ok)

    def test_a_returned_refusal_counts_as_a_refusal(self):
        """The commoner shape: nothing is raised, the result comes back
        carrying error-level issues."""
        session = agent.Session()
        view = dz.inspect(seqio.Record("mystery", "ATGC" * 1000, True))
        session.call("expression.suits_ecoli", view=view)
        self.assertEqual(session.trace.calls[-1].outcome, "declined")

    def test_identical_records_from_different_objects_fingerprint_alike(self):
        a = vector("pET-28a")
        b = vector("pET-28a")
        self.assertEqual(agent._fingerprint(a), agent._fingerprint(b))

    def test_stripping_annotations_is_a_different_question(self):
        a = vector("pET-28a")
        b = seqio.Record(a.name, a.seq, a.circular, [])
        self.assertNotEqual(agent._fingerprint(a), agent._fingerprint(b))

    def test_the_catalogue_lists_every_tool(self):
        text = agent.catalogue()
        for name in agent.REGISTRY:
            self.assertIn(name, text)


# ---------------------------------------------------------------------------
# the invariants
# ---------------------------------------------------------------------------


class TestInvariants(unittest.TestCase):
    def _trace_with(self, *issues):
        t = tracing.Trace()
        t.record(tracing.Call(
            tool="t", layer=1, target=(), tuned=(),
            outcome="returned", issues=tuple(issues),
            numbers=frozenset({1.0, 62.0}),
        ))
        return t

    def test_i1_catches_a_number_no_tool_produced(self):
        t = self._trace_with()
        broken = tracing.check_numbers_traceable(t, "run it at 58 C")
        self.assertTrue(broken)

    def test_i1_accepts_a_number_a_tool_produced(self):
        t = self._trace_with()
        self.assertEqual(
            tracing.check_numbers_traceable(t, "run it at 62 C"), []
        )

    def test_i1_accepts_digit_free_prose(self):
        t = self._trace_with()
        self.assertEqual(
            tracing.check_numbers_traceable(t, "the design succeeded"), []
        )

    def test_i2_catches_a_dropped_issue(self):
        issue = dz.Issue("warning", "the tail contains a second ATG")
        t = self._trace_with(issue)
        broken = tracing.check_issues_survive(t, "all checks passed")
        self.assertTrue(broken)

    def test_i2_catches_a_downgraded_issue(self):
        """The quiet failure: every word true except the level."""
        issue = dz.Issue("error", "the insert is cut by both enzymes")
        t = self._trace_with(issue)
        answer = "  note  the insert is cut by both enzymes"
        self.assertTrue(tracing.check_issues_survive(t, answer))

    def test_i2_accepts_a_wrapped_issue(self):
        issue = dz.Issue("note", "a machine-readable note that is long "
                                 "enough to be wrapped by the renderer")
        t = self._trace_with(issue)
        wrapped = "  note  a machine-readable note that is long\n" \
                  "        enough to be wrapped by the renderer"
        self.assertEqual(tracing.check_issues_survive(t, wrapped), [])

    def test_issues_are_found_at_any_depth(self):
        nested = {"a": [{"b": (dz.Issue("note", "deep"),)}]}
        self.assertEqual(len(tracing.issues_in(nested)), 1)


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator(unittest.TestCase):
    def test_a_full_run_passes_its_own_audit(self):
        out = orchestrate.run(orchestrate.Request(
            insert=str(EXAMPLES / "SOD1.fasta"), vector="pET-28a",
        ))
        self.assertFalse(out.refused)
        self.assertTrue(out.answer)
        self.assertEqual(
            {k: v for k, v in
             tracing.audit(out.trace, out.answer).items() if v}, {}
        )

    def test_every_issue_appears_in_the_answer(self):
        out = orchestrate.run(orchestrate.Request(
            insert=str(EXAMPLES / "SOD1.fasta"), vector="pET-28a",
        ))
        flat = " ".join(out.answer.split())
        for issue in out.trace.issues:
            self.assertIn(" ".join(issue.text.split()), flat)

    def test_the_agent_picks_a_vector_when_none_is_named(self):
        out = orchestrate.run(orchestrate.Request(
            insert=str(EXAMPLES / "SOD1.fasta"), verify=False,
        ))
        self.assertFalse(out.refused)
        self.assertTrue(out.candidates)
        self.assertEqual(out.spec.name, out.candidates[0].spec.name)

    def test_a_mammalian_vector_is_refused_before_any_primer_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mammalian.gb"
            record = seqio.Record(
                "pCMV-ish", "ATGC" * 1200, True,
                [seqio.Feature("CMV promoter", "promoter", 10, 30, True)],
            )
            path.write_text(seqio.write_genbank(record), encoding="utf-8")
            out = orchestrate.run(orchestrate.Request(
                insert=str(EXAMPLES / "SOD1.fasta"), vector=str(path),
            ))
        self.assertTrue(out.refused)
        self.assertIsNone(out.design)
        self.assertNotIn("design.design", [c.tool for c in out.trace.calls])

    def test_the_default_narrator_writes_no_digits(self):
        summary = orchestrate.Summary(
            request=orchestrate.Request(insert="x"), view=None, design=None,
            plan=None, refused=False, refusal="", issues=(),
        )
        prose = orchestrate.plain_narration(summary)
        self.assertEqual(tracing.numbers_in_text(prose), set())

    def test_assemble_refuses_an_answer_with_an_invented_number(self):
        session = agent.Session()
        session.trace.record(tracing.Call(
            tool="t", layer=1, target=(), tuned=(), outcome="returned",
            numbers=frozenset({62.0}),
        ))
        with self.assertRaises(tracing.InvariantError):
            orchestrate.assemble(session, ["use 4 units of enzyme"], "")

    def test_a_standard_route_settles_nothing(self):
        request = orchestrate.Request(insert="x", vector="y")
        self.assertEqual(orchestrate.StandardRoute().route(request), request)


# ---------------------------------------------------------------------------
# reading a sentence
# ---------------------------------------------------------------------------


class TestPlanner(unittest.TestCase):
    def read(self, question: str) -> orchestrate.Request:
        return planner.SentenceRoute(question).route(
            orchestrate.Request(insert="")
        )

    def test_a_plain_request(self):
        request = self.read(
            f"clone {EXAMPLES / 'SOD1.fasta'} into pET-28a"
        )
        self.assertTrue(request.insert.endswith("SOD1.fasta"))
        self.assertEqual(request.vector, "pET-28a")
        self.assertEqual(request.mode, "auto")

    def test_the_organism_is_picked_up(self):
        request = self.read(
            f"clone {EXAMPLES / 'SOD1.fasta'} into pET-28a, a human gene"
        )
        self.assertEqual(request.origin, "human")

    def test_asking_it_to_choose_leaves_the_vector_empty(self):
        request = self.read(
            f"design primers for {EXAMPLES / 'SOD1.fasta'}, "
            "pick the best vector"
        )
        self.assertEqual(request.vector, "")

    def test_a_comparison_is_recognised(self):
        request = self.read(
            f"which vector should I use for {EXAMPLES / 'SOD1.fasta'}?"
        )
        self.assertTrue(request.compare)

    def test_declining_verification_in_several_wordings(self):
        for phrasing in (
            "skip verification",
            "don't bother with sequencing",
            "no colony PCR please",
            "without checking it",
        ):
            request = self.read(
                f"clone {EXAMPLES / 'SOD1.fasta'} into pET-28a, {phrasing}"
            )
            self.assertFalse(request.verify, phrasing)

    def test_a_named_expression_strain_is_used(self):
        request = self.read(
            f"clone {EXAMPLES / 'SOD1.fasta'} into pET-28a in Rosetta(DE3)"
        )
        self.assertEqual(
            strains.get(request.expression_strain).name, "Rosetta(DE3)"
        )

    def test_a_fusion_strategy_is_honoured(self):
        request = self.read(
            f"clone {EXAMPLES / 'SOD1.fasta'} into pET-21b as a "
            "c-terminal fusion"
        )
        self.assertEqual(request.mode, "cfusion")

    def test_an_unnamed_gene_is_refused(self):
        with self.assertRaises(planner.PlannerError):
            self.read("design me some primers")

    def test_an_invented_vector_is_refused(self):
        """A planner may decide; it may not invent. The same gate applies
        whether a model or a regex produced the request."""
        with self.assertRaises(planner.PlannerError):
            planner.validate(orchestrate.Request(
                insert=str(EXAMPLES / "SOD1.fasta"), vector="pUltraExpress-9",
            ))

    def test_an_invented_mode_is_refused(self):
        with self.assertRaises(planner.PlannerError):
            planner.validate(orchestrate.Request(
                insert=str(EXAMPLES / "SOD1.fasta"), mode="superfusion",
            ))

    def test_an_invented_strain_is_refused(self):
        with self.assertRaises(planner.PlannerError):
            planner.validate(orchestrate.Request(
                insert=str(EXAMPLES / "SOD1.fasta"),
                expression_strain="BL99(DE9)",
            ))

    def test_the_request_has_no_numeric_field_for_a_model_to_fill(self):
        """What makes this seam safe: whatever a planner decides, it cannot
        put a number into the answer."""
        import dataclasses

        for field in dataclasses.fields(orchestrate.Request):
            self.assertIn(field.type, ("str", "bool"), field.name)

    def test_the_model_planner_schema_has_no_numbers_either(self):
        for spec in planner.SCHEMA["properties"].values():
            self.assertIn(spec["type"], ("string", "boolean"))

    def test_the_model_planner_validates_what_the_model_said(self):
        """The model's answer is data, not instruction: a vector it made up
        is rejected here, not carried into the pipeline."""

        class FakeBlock:
            type = "text"
            text = json.dumps({
                "insert": "examples/SOD1.fasta",
                "vector": "pDefinitelyReal",
                "mode": "auto",
                "origin": "",
                "verify": True,
                "compare": False,
            })

        class FakeResponse:
            content = [FakeBlock()]

        class FakeMessages:
            def create(self, **_kwargs):
                return FakeResponse()

        class FakeClient:
            messages = FakeMessages()

        route = planner.ClaudePlanner("whatever", client=FakeClient())
        with self.assertRaises(planner.PlannerError):
            route.route(orchestrate.Request(insert=""))

    def test_the_model_planner_accepts_a_valid_answer(self):
        class FakeBlock:
            type = "text"
            text = json.dumps({
                "insert": str(EXAMPLES / "SOD1.fasta"),
                "vector": "pGEX-4T-1",
                "mode": "auto",
                "origin": "human",
                "verify": False,
                "compare": False,
            })

        class FakeResponse:
            content = [FakeBlock()]

        class FakeMessages:
            def create(self, **_kwargs):
                return FakeResponse()

        class FakeClient:
            messages = FakeMessages()

        route = planner.ClaudePlanner("whatever", client=FakeClient())
        request = route.route(orchestrate.Request(insert=""))
        self.assertEqual(request.vector, "pGEX-4T-1")
        self.assertEqual(request.origin, "human")
        self.assertFalse(request.verify)


class TestOrigins(unittest.TestCase):
    def test_a_species_resolves_to_its_clade(self):
        self.assertEqual(source.get("human").name, "mammalian")
        self.assertEqual(source.get("candida").name, "yeast")
        self.assertEqual(source.get("e. coli").name, "bacterial")

    def test_an_unknown_organism_is_refused_helpfully(self):
        with self.assertRaises(KeyError) as caught:
            source.get("tardigrade")
        self.assertIn("mammalian", str(caught.exception))

    def test_a_mammalian_gene_with_rare_codons_asks_for_rosetta(self):
        assessment = source.assess("human", gene("SOD1"))
        self.assertTrue(assessment.wants_rare_codon_strain)


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def run_cli(*argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_no_command_with_no_terminal_prints_help(self):
        """In a pipe or a CI step there is nobody to interview, and a
        blocking prompt there is worse than help text."""
        original = cli.interactive.available
        cli.interactive.available = lambda: False
        try:
            code, text, _ = run_cli()
        finally:
            cli.interactive.available = original
        self.assertEqual(code, 2)
        self.assertIn("clonewright", text)

    def test_no_command_with_a_terminal_starts_the_interview(self):
        original_available = cli.interactive.available
        original_run = cli.interactive.run
        started: list[bool] = []
        cli.interactive.available = lambda: True
        cli.interactive.run = lambda *a, **k: (started.append(True), 0)[1]
        try:
            code, _, _ = run_cli()
        finally:
            cli.interactive.available = original_available
            cli.interactive.run = original_run
        self.assertEqual(code, 0)
        self.assertTrue(started)

    def test_design_exits_zero_on_success(self):
        code, text, _ = run_cli(
            "design", str(EXAMPLES / "SOD1.fasta"),
            "--vector", "pET-28a", "--no-verify",
        )
        self.assertEqual(code, 0)
        self.assertIn("NdeI", text)

    def test_design_exits_one_on_a_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mammalian.gb"
            record = seqio.Record(
                "pCMV-ish", "ATGC" * 1200, True,
                [seqio.Feature("CMV promoter", "promoter", 10, 30, True)],
            )
            path.write_text(seqio.write_genbank(record), encoding="utf-8")
            code, text, _ = run_cli(
                "design", str(EXAMPLES / "SOD1.fasta"), "--vector", str(path),
            )
        self.assertEqual(code, 1)
        self.assertIn("mammalian", text)

    def test_a_missing_file_exits_two(self):
        code, _, err = run_cli("design", "no-such-gene.fasta",
                               "--vector", "pET-28a")
        self.assertEqual(code, 2)
        self.assertIn("no such file", err.lower())

    def test_an_unknown_vector_exits_two(self):
        code, _, err = run_cli("design", str(EXAMPLES / "SOD1.fasta"),
                               "--vector", "pNonsense")
        self.assertEqual(code, 2)
        self.assertIn("pET-28a", err)

    def test_json_output_is_valid_json(self):
        code, text, _ = run_cli(
            "design", str(EXAMPLES / "SOD1.fasta"), "--vector", "pET-28a",
            "--no-verify", "--json",
        )
        self.assertEqual(code, 0)
        payload = json.loads(text)
        self.assertEqual(payload["vector"], "pET-28a(+)")
        self.assertEqual(payload["product"]["n_terminal_addition"],
                         "MGSSHHHHHHSSGLVPRGSHM")
        self.assertIn("design", payload)

    def test_emit_construct_writes_openable_genbank(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "construct.gb"
            code, _, _ = run_cli(
                "design", str(EXAMPLES / "SOD1.fasta"), "--vector", "pET-28a",
                "--no-verify", "--emit-construct", str(out),
            )
            self.assertEqual(code, 0)
            record = seqio.load(out)
        self.assertTrue(record.circular)
        self.assertIn("SOD1", [f.name for f in record.features])

    def test_compare_lists_every_vector(self):
        code, text, _ = run_cli("compare", str(EXAMPLES / "SOD1.fasta"))
        self.assertEqual(code, 0)
        for name in vectors.available():
            self.assertIn(name, text)

    def test_vectors_names_the_bundled_maps(self):
        code, text, _ = run_cli("vectors")
        self.assertEqual(code, 0)
        self.assertIn("pET-28a", text)
        self.assertNotIn("! map file missing", text)

    def test_tools_lists_the_registry(self):
        code, text, _ = run_cli("tools")
        self.assertEqual(code, 0)
        self.assertIn("design.design", text)

    def test_annotate_derives_a_cassette_from_bare_sequence(self):
        bare = vector("pET-28a")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bare.fa"
            source.write_text(f">pET-28a\n{bare.seq}\n", encoding="utf-8")
            out = Path(tmp) / "derived.gb"
            code, text, _ = run_cli("annotate", str(source), "-o", str(out))
            self.assertEqual(code, 0)
            derived = seqio.load(out)
        names = [f.name for f in derived.features]
        self.assertIn("RBS", names)
        self.assertIn("T7 promoter", names)


# ---------------------------------------------------------------------------
# fetching from NCBI
# ---------------------------------------------------------------------------


DATA = ROOT / "tests" / "data"
TRH_URL = (
    "https://www.ncbi.nlm.nih.gov/nuccore/NC_000003.12"
    "?report=genbank&from=129974720&to=129977935"
)
# The protein NCBI annotates for the TRH CDS, NP_009048.1, as the fixture
# carries it. Held here as well so the cross-check is tested against a
# value written down rather than one read out of the same file.
TRH_PROTEIN_LENGTH = 242
TRH_CDS_LENGTH = 729


def fixture(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


class TestLocations(unittest.TestCase):
    def test_a_simple_span(self):
        location = seqio.parse_location("1..20")
        self.assertEqual(location.segments, ((0, 20),))
        self.assertTrue(location.forward)
        self.assertFalse(location.partial)

    def test_complement_flips_the_strand_not_the_bounds(self):
        location = seqio.parse_location("complement(1..20)")
        self.assertEqual(location.segments, ((0, 20),))
        self.assertFalse(location.forward)

    def test_a_join_keeps_its_pieces(self):
        location = seqio.parse_location("join(1098..1308,1980..2497)")
        self.assertEqual(location.segments, ((1097, 1308), (1979, 2497)))
        self.assertEqual(location.spliced_length, TRH_CDS_LENGTH)
        # The outer bounds include the intron, which is the answer a
        # feature table wants and the wrong one for splicing.
        self.assertEqual(location.end - location.start, 1400)

    def test_partial_markers_inside_a_range_still_parse(self):
        """`18..>228` is what NCBI writes for a CDS running off the end of
        a requested region. A pattern that misses it does not give a wrong
        answer, it makes the feature disappear -- in exactly the case
        worth reporting clearly."""
        location = seqio.parse_location("18..>228")
        self.assertIsNotNone(location)
        self.assertEqual(location.segments, ((17, 228),))
        self.assertTrue(location.partial)

    def test_every_partial_spelling(self):
        for raw in ("<1..228", "18..>228", "<1..>228"):
            self.assertTrue(seqio.parse_location(raw).partial, raw)

    def test_extract_joins_then_complements(self):
        # Two codons on the reverse strand: revcomp(ATG+TGG) is CCACAT.
        seq = "CCACAT"
        location = seqio.parse_location("complement(join(1..3,4..6))")
        self.assertEqual(location.extract(seq), "ATGTGG")

    def test_nonsense_is_refused_rather_than_guessed(self):
        self.assertIsNone(seqio.parse_location("somewhere in the middle"))


class TestTargets(unittest.TestCase):
    def test_the_region_url_from_the_nucleotide_viewer(self):
        target = ncbi.parse_target(TRH_URL)
        self.assertEqual(target.accession, "NC_000003.12")
        self.assertEqual((target.start, target.end), (129974720, 129977935))
        self.assertTrue(target.is_region)

    def test_a_url_with_no_region(self):
        target = ncbi.parse_target(
            "https://www.ncbi.nlm.nih.gov/nuccore/NM_007117.5"
        )
        self.assertEqual(target.accession, "NM_007117.5")
        self.assertFalse(target.is_region)

    def test_a_bare_accession(self):
        for text in ("NM_007117.5", "NC_000003.12", "nm_007117"):
            self.assertTrue(ncbi.looks_like_a_target(text), text)
            self.assertEqual(
                ncbi.parse_target(text).accession, text.upper()
            )

    def test_a_file_path_is_not_a_target(self):
        self.assertFalse(ncbi.looks_like_a_target("examples/SOD1.fasta"))
        self.assertFalse(ncbi.looks_like_a_target("ATGGCGACGAAGGCC"))

    def test_a_reversed_view_is_the_minus_strand(self):
        target = ncbi.parse_target(
            "https://www.ncbi.nlm.nih.gov/nuccore/X?from=1&to=9&strand=true"
        )
        self.assertEqual(target.strand, 2)
        self.assertIn("strand=2", ncbi.efetch_url(target))

    def test_positions_are_ordered_and_commas_tolerated(self):
        target = ncbi.parse_target(
            "https://www.ncbi.nlm.nih.gov/nuccore/X?from=9,000&to=1,000"
        )
        self.assertEqual((target.start, target.end), (1000, 9000))

    def test_a_gene_page_says_what_to_paste_instead(self):
        with self.assertRaises(ncbi.FetchError) as caught:
            ncbi.parse_target("https://www.ncbi.nlm.nih.gov/gene/7200")
        self.assertIn("nuccore", str(caught.exception))

    def test_a_non_ncbi_url_is_refused(self):
        with self.assertRaises(ncbi.FetchError):
            ncbi.parse_target("https://example.com/nuccore/NM_007117.5")

    def test_half_a_region_is_refused(self):
        with self.assertRaises(ncbi.FetchError):
            ncbi.parse_target(
                "https://www.ncbi.nlm.nih.gov/nuccore/X?from=100"
            )

    def test_the_fetch_url_carries_the_region_and_identifies_itself(self):
        url = ncbi.efetch_url(ncbi.parse_target(TRH_URL))
        self.assertIn("seq_start=129974720", url)
        self.assertIn("seq_stop=129977935", url)
        self.assertIn("tool=clonewright", url)
        # A chromosome is a CON record: plain `gb` returns no sequence.
        self.assertIn(f"rettype={ncbi.RETTYPE}", url)


class TestSplicing(unittest.TestCase):
    """Against saved records, so these run with no network."""

    def record(self, name: str = "TRH_region.gb") -> ncbi.FetchedRecord:
        return ncbi.parse_record(fixture(name))

    def test_the_record_is_read(self):
        record = self.record()
        self.assertEqual(record.accession, "NC_000003.12")
        self.assertEqual(record.organism, "Homo sapiens")
        self.assertEqual(len(record), 3216)

    def test_the_two_exon_cds_is_found_with_both_exons(self):
        cds = self.record().coding[0]
        self.assertEqual(cds.gene, "TRH")
        self.assertEqual(cds.exons, 2)
        self.assertEqual(cds.location.segments, ((1097, 1308), (1979, 2497)))
        self.assertEqual(cds.length, TRH_CDS_LENGTH)

    def test_the_translation_qualifier_is_joined_without_gaps(self):
        """A /translation is one word wrapped across many lines; joining it
        with spaces would make every comparison fail."""
        cds = self.record().coding[0]
        self.assertEqual(len(cds.translation), TRH_PROTEIN_LENGTH)
        self.assertTrue(cds.translation.startswith("MPGPWLLLALALTLNLTGVPG"))
        self.assertNotIn(" ", cds.translation)

    def test_the_spliced_sequence_is_a_coding_sequence(self):
        record = self.record()
        coding = ncbi.spliced(record.coding[0], record.seq)
        self.assertEqual(len(coding), TRH_CDS_LENGTH)
        self.assertTrue(coding.startswith("ATG"))
        self.assertIn(coding[-3:], ("TAA", "TAG", "TGA"))
        self.assertEqual(len(coding) % 3, 0)

    def test_the_splice_is_confirmed_against_ncbis_own_protein(self):
        record = self.record()
        issues = ncbi.check(record.coding[0], record.seq)
        self.assertEqual([i for i in issues if i.level == "error"], [])
        self.assertTrue(
            any("splice is confirmed" in i.text for i in issues)
        )

    def test_the_cross_check_catches_a_wrong_splice(self):
        """The test the cross-check exists for: move one exon boundary by a
        single base and the sequence still looks like DNA, still amplifies,
        and encodes a different protein."""
        record = self.record()
        original = record.coding[0]
        shifted = dataclasses.replace(
            original,
            location=seqio.Location(
                ((1096, 1308), (1979, 2497)), True, False
            ),
        )
        issues = ncbi.check(shifted, record.seq)
        errors = [i for i in issues if i.level == "error"]
        self.assertTrue(errors)
        self.assertTrue(
            any("does not translate to the protein" in i.text
                for i in errors)
        )

    def test_an_intron_is_reported_as_left_out(self):
        record = self.record()
        issues = ncbi.check(record.coding[0], record.seq)
        intron = next(i for i in issues if "intron" in i.text)
        self.assertIn("671 bp of intron", intron.text)
        self.assertIn("cDNA", intron.text)

    def test_the_mrna_record_gives_the_same_coding_sequence(self):
        """Two independent routes to one gene: a genomic region that has to
        be spliced, and an mRNA whose CDS sits inside the UTRs."""
        genomic = self.record()
        mrna = self.record("TRH_mRNA.gb")
        self.assertEqual(mrna.coding[0].exons, 1)
        self.assertEqual(
            ncbi.spliced(genomic.coding[0], genomic.seq),
            ncbi.spliced(mrna.coding[0], mrna.seq),
        )

    def test_the_cds_is_cut_out_of_the_mrnas_utrs(self):
        mrna = self.record("TRH_mRNA.gb")
        self.assertLess(mrna.coding[0].length, len(mrna))
        self.assertFalse(mrna.seq.startswith("ATG"))
        self.assertTrue(
            ncbi.spliced(mrna.coding[0], mrna.seq).startswith("ATG")
        )

    def test_a_cut_region_is_refused_as_partial(self):
        record = self.record("TRH_cut_region.gb")
        self.assertTrue(record.coding, "the CDS must not disappear")
        cds = record.coding[0]
        self.assertTrue(cds.location.partial)
        errors = [
            i for i in ncbi.check(cds, record.seq) if i.level == "error"
        ]
        self.assertTrue(any("cuts TRH short" in i.text for i in errors))

    def test_a_partial_cds_is_not_blamed_on_the_arithmetic(self):
        record = self.record("TRH_cut_region.gb")
        issues = ncbi.check(record.coding[0], record.seq)
        self.assertFalse(
            any("boundaries or the strand have been read wrongly" in i.text
                for i in issues)
        )

    def test_a_region_with_no_coding_sequence_is_refused(self):
        record = self.record("TRH_no_cds.gb")
        chosen, issues = ncbi.choose(record)
        self.assertIsNone(chosen)
        self.assertTrue(any(i.level == "error" for i in issues))
        self.assertIn("no annotated coding sequence", issues[0].text)

    def test_a_reverse_strand_cds_is_spliced_on_the_right_strand(self):
        """Hand-built, because the fixtures are all forward-strand: the
        exons encode MWK and sit on the minus strand."""
        coding = "ATGTGGAAATAA"
        record_seq = seqio.revcomp(coding)
        text = (
            "LOCUS       TEST                      12 bp    DNA     linear\n"
            "DEFINITION  a minus-strand test case.\n"
            "FEATURES             Location/Qualifiers\n"
            "     CDS             complement(1..12)\n"
            '                     /gene="mwk"\n'
            '                     /translation="MWK"\n'
            "ORIGIN\n"
            f"        1 {record_seq.lower()}\n"
            "//\n"
        )
        record = ncbi.parse_record(text)
        cds = record.coding[0]
        self.assertFalse(cds.location.forward)
        self.assertEqual(ncbi.spliced(cds, record.seq), coding)
        self.assertTrue(
            any("splice is confirmed" in i.text
                for i in ncbi.check(cds, record.seq))
        )


class TestChoosingTheGene(unittest.TestCase):
    def two_genes(self) -> ncbi.FetchedRecord:
        record = ncbi.parse_record(fixture("TRH_region.gb"))
        extra = dataclasses.replace(
            record.coding[0],
            gene="OTHER",
            location=seqio.Location(((0, 300),), True, False),
            translation="",
        )
        record.coding.append(extra)
        return record

    def test_one_coding_sequence_needs_no_decision(self):
        record = ncbi.parse_record(fixture("TRH_region.gb"))
        chosen, issues = ncbi.choose(record)
        self.assertEqual(chosen.gene, "TRH")
        self.assertEqual(issues, [])

    def test_several_takes_the_longest_and_says_so(self):
        chosen, issues = ncbi.choose(self.two_genes())
        self.assertEqual(chosen.gene, "TRH")
        self.assertTrue(
            any(i.level == "warning" and "2 coding sequences" in i.text
                for i in issues)
        )

    def test_a_name_settles_it(self):
        chosen, _ = ncbi.choose(self.two_genes(), gene="OTHER")
        self.assertEqual(chosen.gene, "OTHER")

    def test_a_name_that_matches_nothing_lists_what_is_there(self):
        chosen, issues = ncbi.choose(self.two_genes(), gene="BRCA1")
        self.assertIsNone(chosen)
        self.assertIn("TRH", issues[0].text)

    def test_a_pseudogene_is_not_a_candidate(self):
        record = ncbi.parse_record(fixture("TRH_region.gb"))
        record.coding = [
            dataclasses.replace(record.coding[0], pseudo=True)
        ]
        chosen, issues = ncbi.choose(record)
        self.assertIsNone(chosen)
        self.assertIn("pseudogene", issues[0].text)


class TestFetchCache(unittest.TestCase):
    """The cache is what makes a fetched design reproducible, so it is
    tested as behaviour rather than trusted."""

    def test_a_cached_record_is_read_without_the_network(self):
        target = ncbi.parse_target(TRH_URL)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ncbi.write_cached(target, fixture("TRH_region.gb"), cache)
            text, from_cache = ncbi.fetch(
                target, allow_network=False, cache=cache
            )
        self.assertTrue(from_cache)
        self.assertIn("TRH", text)

    def test_the_cache_name_includes_the_region(self):
        a = ncbi.parse_target(TRH_URL)
        b = ncbi.parse_target(
            "https://www.ncbi.nlm.nih.gov/nuccore/NC_000003.12"
            "?from=1&to=100"
        )
        self.assertNotEqual(ncbi.cache_path(a).name, ncbi.cache_path(b).name)
        self.assertIn("129974720-129977935", ncbi.cache_path(a).name)

    def test_offline_with_a_cold_cache_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ncbi.FetchError) as caught:
                ncbi.fetch(
                    ncbi.parse_target(TRH_URL),
                    allow_network=False, cache=Path(tmp),
                )
        self.assertIn("network is switched off", str(caught.exception))

    def test_a_truncated_cache_entry_is_ignored(self):
        target = ncbi.parse_target(TRH_URL)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            ncbi.cache_path(target, cache).parent.mkdir(
                parents=True, exist_ok=True
            )
            ncbi.cache_path(target, cache).write_text(
                "LOCUS  half a file", encoding="utf-8"
            )
            self.assertIsNone(ncbi.read_cached(target, cache))


class TestFetchedDesign(unittest.TestCase):
    """The whole pipeline from a URL, served from a primed cache."""

    def run_from_url(self, url: str, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for name, source in (
                (TRH_URL, "TRH_region.gb"),
                ("NM_007117.5", "TRH_mRNA.gb"),
            ):
                ncbi.write_cached(
                    ncbi.parse_target(name), fixture(source), cache
                )
            original = ncbi.default_cache
            ncbi.default_cache = lambda: cache
            try:
                return orchestrate.run(orchestrate.Request(
                    insert=url, allow_network=False, **kwargs
                ))
            finally:
                ncbi.default_cache = original

    def test_a_url_designs_end_to_end_and_passes_the_audit(self):
        out = self.run_from_url(TRH_URL, vector="pET-28a", verify=False)
        self.assertFalse(out.refused)
        self.assertEqual(
            {k: v for k, v in
             tracing.audit(out.trace, out.answer).items() if v}, {}
        )

    def test_the_fetch_panel_reports_the_exons_and_the_confirmation(self):
        out = self.run_from_url(TRH_URL, vector="pET-28a", verify=False)
        self.assertIn("1098-1308, 1980-2497", out.answer)
        self.assertIn("splice is confirmed", out.answer)
        self.assertIn("NP_009048.1", out.answer)

    def test_the_gene_name_comes_from_the_record(self):
        out = self.run_from_url(TRH_URL, vector="pET-28a", verify=False)
        self.assertIn("TRH_F", out.answer)
        self.assertEqual(out.design.insert.name, "TRH")

    def test_the_insert_is_the_spliced_cds_not_the_region(self):
        out = self.run_from_url(TRH_URL, vector="pET-28a", verify=False)
        self.assertEqual(len(out.design.insert), TRH_CDS_LENGTH)

    def test_an_accession_works_as_well_as_a_url(self):
        out = self.run_from_url("NM_007117.5", vector="pET-28a",
                                verify=False)
        self.assertFalse(out.refused)
        self.assertEqual(len(out.design.insert), TRH_CDS_LENGTH)

    def test_it_can_choose_a_vector_for_a_fetched_gene(self):
        out = self.run_from_url(TRH_URL, verify=False)
        self.assertFalse(out.refused)
        self.assertTrue(out.candidates)

    def test_a_fetch_that_cannot_be_served_refuses_rather_than_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ncbi.default_cache
            ncbi.default_cache = lambda: Path(tmp)
            try:
                out = orchestrate.run(orchestrate.Request(
                    insert=TRH_URL, vector="pET-28a", allow_network=False,
                ))
            finally:
                ncbi.default_cache = original
        self.assertTrue(out.refused)
        self.assertIsNone(out.design)
        self.assertIn("network is switched off", out.answer)


# ---------------------------------------------------------------------------
# the interactive session
# ---------------------------------------------------------------------------


SOD1 = str(EXAMPLES / "SOD1.fasta")


class TestInteractive(unittest.TestCase):
    """Driven from a list of answers, which is the only way an interactive
    program gets tested: `Console` injects input and output rather than
    calling them."""

    def session(self, answers: list[str]) -> tuple[int, str]:
        console, printed = interactive.Console.scripted(answers)
        code = interactive.run(console)
        return code, "\n".join(printed)

    def test_the_shortest_complete_session(self):
        code, text = self.session([SOD1, "1", "", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("NdeI", text)
        self.assertIn("pET-28a", text)

    def test_defaults_alone_produce_a_design(self):
        """Enter at every question has to work, or the defaults are a lie."""
        code, text = self.session([SOD1, "", "", "", "n"])
        self.assertEqual(code, 0)
        self.assertIn("ranked by designing into each one", text)

    def test_it_says_what_it_read(self):
        _code, text = self.session([SOD1, "1", "", "n", "n"])
        self.assertIn("read SOD1, 465 bp", text)
        self.assertIn("reads cleanly from a start codon", text)

    def test_a_bad_path_is_re_asked_not_fatal(self):
        code, text = self.session(["no-such-gene.fa", SOD1, "1", "", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("no such file", text)
        self.assertIn("read SOD1", text)

    def test_a_pasted_sequence_works_and_can_be_named(self):
        sequence = gene("SOD1").seq
        code, text = self.session([sequence, "MyGene", "1", "", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("MyGene_F", text)
        self.assertNotIn("<literal>_F", text)

    def test_a_number_outside_the_menu_is_re_asked(self):
        code, text = self.session([SOD1, "17", "2", "", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("pick a number between 1 and", text)
        self.assertIn("pET-21b", text)

    def test_an_unknown_organism_is_re_asked(self):
        code, text = self.session([SOD1, "1", "tardigrade", "yeast", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("unknown source organism", text)
        self.assertIn("yeast or other fungus", text)

    def test_a_vector_of_your_own_is_accepted_by_path(self):
        code, text = self.session([
            SOD1, "6", str(MAPS / "pGEX-4T-1.gb"), "", "n", "n",
        ])
        self.assertEqual(code, 0)
        self.assertIn("pGEX-4T-1", text)

    def test_a_bad_vector_path_is_re_asked(self):
        code, text = self.session([
            SOD1, "6", "nowhere.gb", str(MAPS / "pET-28a.gb"),
            "", "n", "n",
        ])
        self.assertEqual(code, 0)
        self.assertIn("neither a file nor a vector", text)

    def test_a_yes_or_no_question_rejects_anything_else(self):
        code, text = self.session([SOD1, "1", "", "maybe", "n", "n"])
        self.assertEqual(code, 0)
        self.assertIn("please answer y or n", text)

    def test_verification_can_be_declined(self):
        _code, with_it = self.session([SOD1, "1", "", "y", "n"])
        _code, without = self.session([SOD1, "1", "", "n", "n"])
        self.assertIn("colony PCR screen", with_it)
        self.assertNotIn("colony PCR screen", without)

    def test_running_out_of_answers_stops_cleanly(self):
        code, text = self.session([SOD1])
        self.assertEqual(code, 130)
        self.assertIn("stopped.", text)

    def test_it_can_write_the_construct(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "built.gb"
            code, text = self.session([
                SOD1, "1", "", "n", "y", str(out),
            ])
            self.assertEqual(code, 0)
            self.assertIn("written:", text)
            record = seqio.load(out)
        self.assertTrue(record.circular)
        self.assertIn("SOD1", [f.name for f in record.features])

    def test_a_refused_design_exits_one_and_says_it_is_a_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mammalian.gb"
            record = seqio.Record(
                "pCMV-ish", "ATGC" * 1200, True,
                [seqio.Feature("CMV promoter", "promoter", 10, 30, True)],
            )
            path.write_text(seqio.write_genbank(record), encoding="utf-8")
            code, text = self.session([SOD1, "6", str(path), "", "n"])
        self.assertEqual(code, 1)
        self.assertIn("mammalian", text)
        self.assertIn("refusal, not a crash", text)

    def primed_cache(self, tmp: str) -> None:
        """Fill a cache so the session fetches nothing."""
        for name, source in (
            (TRH_URL, "TRH_region.gb"),
            ("XX_000001.1", "two_genes.gb"),
        ):
            ncbi.write_cached(
                ncbi.parse_target(name), fixture(source), Path(tmp)
            )

    def offline_session(self, tmp: str, answers: list[str]):
        self.primed_cache(tmp)
        original = ncbi.default_cache
        ncbi.default_cache = lambda: Path(tmp)
        try:
            console, printed = interactive.Console.scripted(answers)
            code = interactive.run(console)
        finally:
            ncbi.default_cache = original
        return code, "\n".join(printed)

    def test_an_ncbi_link_is_fetched_and_spliced(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.offline_session(
                tmp, [TRH_URL, "1", "", "n", "n"]
            )
        self.assertEqual(code, 0)
        self.assertIn("fetching from NCBI", text)
        self.assertIn("splice is confirmed", text)
        self.assertIn("read TRH, 729 bp", text)
        self.assertIn("TRH_F", text)

    def test_a_region_with_two_genes_asks_which_one(self):
        """The longest-wins rule is a guess, and the person looking at the
        page knows which gene they linked."""
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.offline_session(
                tmp, ["XX_000001.1", "1", "6", str(MAPS / "pET-28a.gb"),
                      "", "n", "n"],
            )
        self.assertIn("2 coding sequences", text)
        self.assertIn("SHORT", text)
        self.assertIn("LONGER", text)
        # Answering 1 takes SHORT, not the longest.
        self.assertIn("read SHORT", text)
        self.assertIn("12 bp", text)
        self.assertEqual(code, 1)  # a 12 bp insert is refused downstream

    def test_a_link_that_cannot_be_fetched_is_re_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.offline_session(
                tmp,
                ["https://www.ncbi.nlm.nih.gov/gene/7200", SOD1,
                 "1", "", "n", "n"],
            )
        self.assertEqual(code, 0)
        self.assertIn("gene page", text)
        self.assertIn("read SOD1", text)

    def test_the_interactive_subcommand_runs_the_same_thing(self):
        original = cli.interactive.run
        seen: list[object] = []
        cli.interactive.run = lambda *a, **k: (seen.append(True), 0)[1]
        try:
            code, _, _ = run_cli("interactive")
        finally:
            cli.interactive.run = original
        self.assertEqual(code, 0)
        self.assertTrue(seen)

    def test_every_catalog_vector_offers_a_short_summary_for_the_menu(self):
        for name in vectors.available():
            self.assertTrue(vectors.CATALOG[name].summary, name)
            self.assertLess(len(vectors.CATALOG[name].summary), 50, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
