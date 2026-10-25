"""The finished plasmid, as a molecule you can open rather than a string.

`Design.construct` is the right answer to "what did we build" and the wrong
shape to look at: a few thousand bases of text with nothing marked on it.
This module carries the vector's own annotations across the ligation,
adds the insert and the two regenerated sites, and hands back a `Record`
that opens in SnapGene or Benchling with the cassette drawn where it
belongs.

The arithmetic is the part worth being careful about.  The construct is the
backbone followed by the digested insert, where the backbone is the vector
read from the 3' cut round to the 5' cut -- so a vector feature's new
position is its old one measured from the 3' cut, and a feature the
excision removed is one whose span runs off the end of the backbone.  That
second case is the whole reason this is not a simple offset: a map that
confidently draws a feature which is no longer in the plasmid is worse
than a map with nothing drawn on it.
"""

from __future__ import annotations

from .seqio import Feature, Record


def assemble(design, *, name: str = "") -> Record:
    """The ligated plasmid, circular, with every feature that survived.

    Features are carried across in the design's own frame, which may be
    the reverse complement of the map that went in -- `Record.reverse_
    complement` brings annotations with it, so they are already where this
    needs them.
    """
    vector = design.vector
    n_vector = len(vector)
    cut5 = design.pair.pos_five + design.pair.five.top
    cut3 = design.pair.pos_three + design.pair.three.top
    backbone_len = len(design.backbone)
    sequence = design.construct

    features: list[Feature] = []
    for f in vector.features:
        start = (f.start - cut3) % n_vector
        end = start + len(f)
        if end > backbone_len:
            # Either the excision removed it, or it straddles a cut and
            # only part of it is left.  Neither is a feature of this
            # plasmid, so neither is drawn on it.
            continue
        features.append(Feature(f.name, f.kind, start, end, f.forward))

    insert_start = backbone_len
    fragment = len(design.insert_fragment)
    features.append(Feature(
        design.insert.name, "CDS", insert_start, insert_start + fragment, True
    ))

    # The two sites, as they are regenerated in the product.  Worth
    # drawing because they are what the diagnostic digest cuts, and
    # because seeing them on the map is how you notice a site that the
    # ligation did not actually restore.
    five, three = design.pair.five, design.pair.three
    lead = design.forward.site[five.top :] + design.forward.pad
    features.append(Feature(
        f"{five.name} (5' junction)", "misc_feature",
        max(insert_start - five.top, 0), insert_start + len(lead), True,
    ))
    features.append(Feature(
        f"{three.name} (3' junction)", "misc_feature",
        max(len(sequence) - len(three.site), 0), len(sequence), True,
    ))

    features.sort(key=lambda f: (f.start, f.end))
    return Record(
        name=name or f"{design.insert.name}_in_{design.vector.name}",
        seq=sequence,
        circular=True,
        features=features,
    )
