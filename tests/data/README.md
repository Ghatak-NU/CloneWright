# Test fixtures

GenBank records saved from NCBI, used so the NCBI tests run offline and
deterministically. RefSeq records are works of the US government and are
in the public domain.

| File | What it is | What it tests |
|---|---|---|
| `TRH_region.gb` | `NC_000003.12` 129974720-129977935 | A two-exon CDS, `join(1098..1308,1980..2497)`: the splice, and the cross-check against NCBI's own `/translation` |
| `TRH_mRNA.gb` | `NM_007117.5` | The same gene as an mRNA, where the CDS is one span inside the UTRs |
| `TRH_cut_region.gb` | `NC_000003.12` 129975800-129976400 | A CDS the region cuts in half, written `18..>228` -- the partial marker that must not make the feature disappear |
| `TRH_no_cds.gb` | `NC_000003.12` 129974720-129975000 | A region with no coding sequence in it |

Each was fetched with the URL form the module accepts, and the file is the
response body unchanged. To refresh one, delete it and run the test suite
with the network available, or fetch it by hand with
`clonewright design <url>` and copy it out of the cache
(`~/.clonewright/ncbi`, or `$CLONEWRIGHT_CACHE`).
