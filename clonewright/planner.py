"""Turning a sentence into a `Request`: the one judgement a model is trusted with.

*"Clone SOD1 into pET-28a, it's a human gene, and tell me how to check it"*
is not a question code can parse in general, and it is the question people
actually ask.  So this module fills the `Planner` seam in `orchestrate`,
twice:

* `SentenceRoute` reads the sentence with patterns.  It needs no network, no
  API key and no model, it is tested, and it covers the shapes that come up
  in practice.  It is the default.
* `ClaudePlanner` asks Claude.  It handles the sentences patterns do not,
  and it is optional in every sense -- an extra dependency, an extra
  credential, and never reached unless asked for.

**Why a model can be trusted with this and nothing else.** Look at what a
planner can return: `Request` holds a path, names from fixed sets, and
flags.  There is not one numeric field on it.  So no matter what a model
decides here, it cannot put a melting temperature, a fragment size or a
volume into the answer -- those come from Layer 1, and `trace.audit`
rejects an answer carrying a number no tool produced.  The seam is narrow
by construction rather than by good behaviour.

Everything a model returns is then validated against the catalog, the mode
list and the strain table before it reaches the pipeline.  A model's output
is data, not instruction: a vector name that is neither a bundled vector
nor a file on disk is rejected here rather than carried forward.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

from . import source as source_mod
from . import strains as strains_mod
from . import vectors as vectors_mod
from .orchestrate import Request

MODES = ("auto", "native", "nfusion", "cfusion")

# Words that name an organism, as people write them.  The canonical names
# and the species aliases both live in `source`, so this reads them from
# there instead of keeping a second list that can drift out of step with
# the one that actually resolves.
ORGANISM_WORDS: dict[str, str] = {
    **{name: name for name in source_mod.ORIGINS},
    **{alias: alias for alias in source_mod.ALIASES},
}


class PlannerError(ValueError):
    """Raised when a sentence cannot be turned into a request."""


# Words that are never a gene, however they sit in the sentence.  Short and
# deliberately incomplete: the test below is the real filter, and this only
# removes the words that would otherwise pass it.
_FILLER = frozenset({
    "me", "my", "some", "a", "an", "the", "it", "this", "that", "those",
    "primers", "primer", "oligos", "oligo", "something", "anything", "for",
    "into", "in", "to", "and", "with", "please", "up", "out", "best",
    "vector", "vectors", "gene", "sequence", "construct", "plasmid",
})


def _looks_like_a_gene(token: str) -> bool:
    """Whether a word could be a gene name or a sequence file.

    Three ways to qualify: it is a file that exists, it has a sequence
    file's extension, or it reads like a gene symbol -- which in practice
    means it carries a capital or a digit.  `SOD1`, `TNFRSF10A` and
    `rad26` pass; `me`, `some` and `please` do not.
    """
    if not token or token.lower() in _FILLER:
        return False
    if Path(token).exists():
        return True
    if re.search(r"\.(?:fa|fasta|gb|gbk|dna|txt|seq)$", token, re.I):
        # A named file that is not there is a missing file, not a missing
        # gene: let `seqio.load` say "no such file", which is the useful
        # message, rather than refusing here as if nothing was named.
        return True
    if len(token) < 3:
        return False
    return bool(re.search(r"[A-Z]", token) or re.search(r"\d", token))


# ---------------------------------------------------------------------------
# validation: the same gate for both planners
# ---------------------------------------------------------------------------


def validate(request: Request) -> Request:
    """Check every field against the fixed sets it is drawn from.

    Applied to whatever a planner returns, model or not.  A planner is
    allowed to decide; it is not allowed to invent a vector, a mode or a
    strain, and the difference is enforced here rather than discovered
    three tools later.
    """
    if not request.insert:
        raise PlannerError("no gene named: say which sequence to clone")

    if request.mode not in MODES:
        raise PlannerError(
            f"{request.mode!r} is not a cloning mode; the modes are "
            + ", ".join(MODES)
        )

    if request.vector:
        if (
            vectors_mod.spec_for(request.vector) is None
            and not Path(request.vector).exists()
        ):
            raise PlannerError(
                f"{request.vector!r} is neither a bundled vector nor a file. "
                f"Bundled: {', '.join(vectors_mod.CATALOG)}"
            )

    for field, value in (
        ("cloning_strain", request.cloning_strain),
        ("expression_strain", request.expression_strain),
    ):
        try:
            strains_mod.get(value)
        except KeyError as exc:
            raise PlannerError(f"{field}: {exc}") from exc

    if request.origin:
        try:
            source_mod.get(request.origin)
        except KeyError as exc:
            raise PlannerError(str(exc)) from exc
    return request


# ---------------------------------------------------------------------------
# the pattern planner
# ---------------------------------------------------------------------------


class SentenceRoute:
    """Reads a request out of a sentence with patterns, and no model.

    The default, and not a placeholder for the other one.  The sentences
    this has to handle are short and shaped alike -- a gene, a vector, an
    organism, sometimes a mode -- and a regular expression that gets them
    right is better than a model call that gets them right: it is free,
    offline, deterministic and testable.
    """

    def __init__(self, question: str = ""):
        self.question = question

    def route(self, request: Request) -> Request:
        if not self.question:
            return validate(request)
        return validate(self.read(self.question, request))

    def read(self, question: str, base: Request | None = None) -> Request:
        text = question.strip()
        low = text.lower()
        request = base or Request(insert="")

        insert = request.insert
        vector = request.vector
        mode = request.mode
        origin = request.origin
        verify = request.verify
        compare = request.compare

        # "into <vector>" / "in <vector>" / "using <vector>"
        into = re.search(
            r"\b(?:into|in to|in|using|with)\s+([\w./\\()+-]+)", text, re.I
        )
        if into:
            candidate = into.group(1).rstrip(".,;")
            if (
                vectors_mod.spec_for(candidate) is not None
                or Path(candidate).exists()
            ):
                vector = candidate

        # "clone X", "express X", or the first thing that looks like a file.
        path = re.search(r"\b([\w./\\-]+\.(?:fa|fasta|gb|gbk|dna|txt|seq))\b",
                         text, re.I)
        if path:
            insert = path.group(1)
        else:
            # Every word after the verb, in order, until one looks like a
            # gene.  Taking the *first* word was wrong: "design me some
            # primers" named no gene and produced `insert="me"`, which is
            # a planner inventing a target out of filler -- the one thing
            # this seam must not do. Nothing found means nothing named,
            # and `validate` turns that into a refusal.
            after = re.search(
                r"\b(?:clone|express|design|make|order|put|insert|"
                r"primers?\s+for)\b(.*)", text, re.I
            )
            if after:
                for token in re.findall(r"[\w./\\-]+", after.group(1)):
                    if _looks_like_a_gene(token):
                        insert = token.rstrip(".,;")
                        break

        for word, canonical in ORGANISM_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", low):
                origin = canonical
                break

        for candidate in ("nfusion", "cfusion", "native"):
            if candidate in low:
                mode = candidate
        if "n-terminal" in low or "n terminal" in low:
            mode = "nfusion"
        elif "c-terminal" in low or "c terminal" in low:
            mode = "cfusion"

        # "no verification", "skip the sequencing", "don't bother with
        # checking it" -- the filler between the negation and the thing
        # being refused is what varies, so a couple of words of it are
        # allowed through.
        if re.search(
            r"\b(?:don't|do not|no|skip|without|forget)\s+"
            r"(?:\w+\s+){0,3}?(?:verif|sequenc|check|screen|colony)",
            low,
        ):
            verify = False
        if re.search(r"\b(?:compare|which vector|best vector|all vectors|"
                     r"rank)\b", low):
            compare = True
        if re.search(r"\bpick (?:a|the best) vector\b", low):
            vector = ""

        # A named strain anywhere in the sentence.
        expression_strain = request.expression_strain
        for key in strains_mod.STRAINS:
            strain = strains_mod.STRAINS[key]
            if strain.name.lower() in low and strain.role != "cloning":
                expression_strain = key

        return replace(
            request,
            insert=insert,
            vector=vector,
            mode=mode,
            origin=origin,
            verify=verify,
            compare=compare,
            expression_strain=expression_strain,
        )


# ---------------------------------------------------------------------------
# the model planner
# ---------------------------------------------------------------------------


MODEL = "claude-opus-5-5"

SYSTEM = """\
You turn a molecular biologist's sentence into a job description for a \
primer design tool. You do not design anything and you do not do any \
arithmetic: you only say what was asked for.

Rules:
- `insert` is the gene: copy the file path or gene name exactly as written.
- `vector` is one of the names you are given, or a file path the sentence \
contains. If the sentence does not name one, or asks the tool to choose, \
leave it empty.
- `mode` is `auto` unless the sentence insists on a particular cloning \
strategy.
- `origin` is the organism the gene came from, only if the sentence says.
- Never guess. An empty field means "not stated", which the tool handles; \
a guessed field is a wrong answer it cannot detect.
"""

# No numeric field anywhere, which is what makes this seam safe: whatever
# the model decides, it cannot put a number into the finished answer.
SCHEMA = {
    "type": "object",
    "properties": {
        "insert": {"type": "string"},
        "vector": {"type": "string"},
        "mode": {"type": "string", "enum": list(MODES)},
        "origin": {"type": "string"},
        "verify": {"type": "boolean"},
        "compare": {"type": "boolean"},
    },
    "required": ["insert", "vector", "mode", "origin", "verify", "compare"],
    "additionalProperties": False,
}


class ClaudePlanner:
    """Asks Claude what was asked for, then validates the answer.

    Optional: `pip install clonewright[agent]` and a credential in the
    environment.  Without either, this raises and the caller falls back to
    `SentenceRoute`, which is why nothing in the package depends on it.
    """

    def __init__(self, question: str, *, model: str = MODEL, client=None):
        self.question = question
        self.model = model
        self._client = client

    def client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise PlannerError(
                "the Claude planner needs the anthropic package: "
                "pip install 'clonewright[agent]'. Without it, "
                "SentenceRoute reads the sentence with patterns instead"
            ) from exc
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise PlannerError(
                "no Anthropic credential in the environment; set "
                "ANTHROPIC_API_KEY or run `ant auth login`"
            )
        self._client = anthropic.Anthropic()
        return self._client

    def route(self, request: Request) -> Request:
        bundled = ", ".join(vectors_mod.CATALOG)
        response = self.client().messages.create(
            model=self.model,
            max_tokens=8192,
            system=SYSTEM,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": (
                    f"Bundled vectors: {bundled}\n\n"
                    f"The sentence: {self.question}"
                ),
            }],
        )
        text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        try:
            fields = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlannerError(
                f"the planner did not return readable JSON: {exc}"
            ) from exc

        # Everything the model said, checked against the fixed sets before
        # it goes anywhere near a tool.
        return validate(replace(
            request,
            insert=str(fields.get("insert") or request.insert),
            vector=str(fields.get("vector") or ""),
            mode=str(fields.get("mode") or "auto"),
            origin=ORGANISM_WORDS.get(
                str(fields.get("origin") or "").strip().lower(), ""
            ),
            verify=bool(fields.get("verify", True)),
            compare=bool(fields.get("compare", False)),
        ))


def for_question(question: str, *, use_model: bool = False):
    """The planner to use for one sentence."""
    if use_model:
        return ClaudePlanner(question)
    return SentenceRoute(question)
