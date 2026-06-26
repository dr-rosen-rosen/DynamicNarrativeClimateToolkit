"""
linguistic_features.py
----------------------
Per-sentence and per-report lexical and syntactic features for safety incident
narratives, derived from spaCy's POS tagger, morphologizer, and dependency
parser. No LIWC; Pennebaker-style dimensions are approximated from POS tags and
small curated lexicons and are clearly labelled as approximations.

Design summary (see linguistic_features_design.md):
- Five feature families: A passive voice, B function-word/style, C tense/aspect,
  D POS rates + lexical diversity, E syntactic complexity, plus optional
  F Pennebaker composites.
- Two granularity tiers:
    * Tier 1 (per-sentence): passive, style, tense, POS rates, and the
      per-sentence structural values (mean_sent_len, dep distance, tree depth,
      clause density, subordination). These aggregate to report level via
      mean / length_weighted / climate_weighted and are climate-weightable.
    * Tier 2 (report-level only): lexical diversity (TTR / MATTR). These need a
      long token span to be meaningful, are computed once over the whole report,
      and are NOT climate-weightable.
- Conventions mirror temporal_features.py: dataclass config, pure compute
  functions on a parsed Doc/Span, an optional stateful extractor class, no
  hidden global state. Missing/undefined values are np.nan, never 0.

Caveats baked into the data (see preprocessing audit):
- NRC stored narratives are lowercased and word-glued at scrape time; do not
  compute syntactic features from them until re-scraped with original case.
- Rail (FRA) narratives are ~100% all-caps in the source; tagger/parser
  accuracy degrades. Aviation (ASRS) is mostly clean mixed-case.
Do not lowercase text before parsing: it kills proper-noun detection and
degrades tagging/parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:  # spaCy is a project dependency; import lazily-friendly for type checking.
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
except Exception:  # pragma: no cover - spaCy always present in the venv.
    spacy = None  # type: ignore
    Language = object  # type: ignore
    Doc = object  # type: ignore
    Span = object  # type: ignore


AggMethod = Literal[
    "mean", "length_weighted", "climate_weighted", "climate_top_k", "climate_threshold"
]

# Lexicons for function-word / style features (Family B). Lowercased lemmas.
_PRON_I = {"i", "me", "my", "mine", "myself"}
_PRON_WE = {"we", "us", "our", "ours", "ourselves"}
_PRON_YOU = {"you", "your", "yours", "yourself", "yourselves"}
_PRON_3RD = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
}
_ARTICLES = {"a", "an", "the"}
_NEG_LEMMAS = {"not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor"}
_QUANT = {"all", "any", "some", "many", "few", "several", "most", "both", "each", "every"}

# Dependency labels that mark a passive construction (Family A).
_PASSIVE_DEPS = {"nsubjpass", "auxpass", "csubjpass"}
# Dependency labels that mark a clausal subject (used to count finite clauses).
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


@dataclass
class LinguisticConfig:
    # Which feature families to compute.
    compute_passive: bool = True       # Family A
    compute_style: bool = True         # Family B
    compute_tense: bool = True         # Family C
    compute_pos_rates: bool = True     # Family D part 1
    compute_diversity: bool = True     # Family D part 2 (TTR, MATTR) - report tier
    compute_complexity: bool = True    # Family E
    compute_composites: bool = False   # Family F (off by default)

    # Future-tense heuristic (Family C) is noisy in English; gate it separately.
    compute_future: bool = True

    # Stability floors (token counts below which a feature is NaN, not computed).
    min_tokens_for_complexity: int = 20
    min_tokens_for_diversity: int = 20
    min_sent_tokens: int = 3           # skip very short sentences in complexity
    mattr_window: int = 50

    # spaCy model. NER is disabled at load (not needed for any feature).
    spacy_model: str = "en_core_web_sm"

    # Keep raw counts alongside rates (useful for downstream re-aggregation).
    include_counts: bool = True


@dataclass
class AggregationConfig:
    method: AggMethod = "length_weighted"
    k: Optional[int] = None             # climate_top_k
    threshold: Optional[float] = None   # climate_threshold
    # If True, treat the weights as already-normalized; else normalize to sum 1.
    normalize_weights: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_rate(n: float, d: float) -> float:
    """n / d, returning NaN when the denominator is zero (undefined, not 0)."""
    if d is None or d == 0:
        return np.nan
    return n / d


def _finite_verbs(span) -> List:
    """Tokens that are finite verbs (VERB/AUX with VerbForm=Fin)."""
    out = []
    for tok in span:
        if tok.pos_ in ("VERB", "AUX"):
            vf = tok.morph.get("VerbForm")
            if "Fin" in vf:
                out.append(tok)
    return out


def _iter_sents(doc_or_span):
    """Yield sentences; works on a Doc (has .sents) or a single-sentence Span."""
    sents = getattr(doc_or_span, "sents", None)
    if sents is None:
        return [doc_or_span]
    try:
        return list(sents)
    except ValueError:
        # No sentence boundaries set (parser/senter disabled): treat as one.
        return [doc_or_span]


# ---------------------------------------------------------------------------
# Family A - passive voice
# ---------------------------------------------------------------------------

def passive_features(doc, include_counts: bool = True) -> Dict[str, float]:
    """
    Passive-voice features. A clause is passive iff its head verb has a child
    with dep_ in {nsubjpass, auxpass, csubjpass}. passive_n counts distinct
    passive head verbs (one per passive clause).
    """
    n_tokens = len(doc)
    sents = _iter_sents(doc)
    n_sents = len(sents)

    passive_heads = set()
    for tok in doc:
        if tok.dep_ in _PASSIVE_DEPS:
            passive_heads.add(tok.head.i)
    passive_n = len(passive_heads)

    n_clauses = sum(1 for tok in doc if tok.dep_ in _SUBJECT_DEPS)

    out: Dict[str, float] = {
        "passive_per_sent": _safe_rate(passive_n, n_sents),
        "passive_per_clause": _safe_rate(passive_n, n_clauses),
        "passive_rate_tokens": _safe_rate(passive_n, n_tokens),
    }
    if include_counts:
        out["passive_n"] = float(passive_n)
    return out


# ---------------------------------------------------------------------------
# Family B - function-word / style (Pennebaker core, POS-derived)
# ---------------------------------------------------------------------------

def style_features(doc, include_counts: bool = True) -> Dict[str, float]:
    """Function-word and style rates per token, with raw counts retained."""
    n = len(doc)
    counts = {
        "pron_i": 0, "pron_we": 0, "pron_you": 0, "pron_3rd": 0,
        "article": 0, "prep": 0, "cconj": 0, "sconj": 0,
        "aux": 0, "modal": 0, "neg": 0, "quant": 0,
    }
    for tok in doc:
        lemma = tok.lemma_.lower()
        if tok.tag_ in ("PRP", "PRP$"):
            if lemma in _PRON_I:
                counts["pron_i"] += 1
            elif lemma in _PRON_WE:
                counts["pron_we"] += 1
            elif lemma in _PRON_YOU:
                counts["pron_you"] += 1
            elif lemma in _PRON_3RD:
                counts["pron_3rd"] += 1
        # 3rd-person possessives/pronouns may also arrive without PRP tag.
        elif lemma in _PRON_3RD and tok.pos_ == "PRON":
            counts["pron_3rd"] += 1

        if tok.pos_ == "DET" and lemma in _ARTICLES:
            counts["article"] += 1
        if tok.pos_ == "ADP":
            counts["prep"] += 1
        if tok.pos_ == "CCONJ":
            counts["cconj"] += 1
        if tok.pos_ == "SCONJ":
            counts["sconj"] += 1
        if tok.pos_ == "AUX":
            counts["aux"] += 1
        if tok.tag_ == "MD":
            counts["modal"] += 1
        if tok.dep_ == "neg" or lemma in _NEG_LEMMAS:
            counts["neg"] += 1
        if lemma in _QUANT:
            counts["quant"] += 1

    out: Dict[str, float] = {f"{k}_rate": _safe_rate(v, n) for k, v in counts.items()}
    if include_counts:
        out.update({f"{k}_n": float(v) for k, v in counts.items()})
    return out


# ---------------------------------------------------------------------------
# Family C - tense and aspect
# ---------------------------------------------------------------------------

def tense_features(doc, compute_future: bool = True,
                   include_counts: bool = True) -> Dict[str, float]:
    """
    Tense/aspect rates over finite verbs. Undefined (NaN) when the span has no
    finite verbs - common for noun-fragment log entries.
    """
    fin = _finite_verbs(doc)
    n_fin = len(fin)

    past = pres = prog = perf = 0
    for tok in fin:
        tense = tok.morph.get("Tense")
        aspect = tok.morph.get("Aspect")
        if "Past" in tense:
            past += 1
        if "Pres" in tense:
            pres += 1
        if "Prog" in aspect:
            prog += 1
        if "Perf" in aspect:
            perf += 1

    out: Dict[str, float] = {
        "past_rate": _safe_rate(past, n_fin),
        "pres_rate": _safe_rate(pres, n_fin),
        "progressive_rate": _safe_rate(prog, n_fin),
        "perfect_rate": _safe_rate(perf, n_fin),
    }

    if compute_future:
        # Heuristic: modal 'will'/'shall' governing a VERB, or 'going to' + VERB.
        future = 0
        for tok in doc:
            lemma = tok.lemma_.lower()
            if tok.tag_ == "MD" and lemma in ("will", "shall"):
                if tok.head.pos_ == "VERB" or any(c.pos_ == "VERB" for c in tok.head.children):
                    future += 1
            elif lemma == "go" and "Prog" in tok.morph.get("Aspect"):
                # 'going to <verb>' - look for an xcomp/advcl 'to' + VERB child.
                if any(c.pos_ == "VERB" for c in tok.children):
                    future += 1
        out["future_rate"] = _safe_rate(future, n_fin)
        if include_counts:
            out["future_n"] = float(future)

    if include_counts:
        out.update({
            "fin_verb_n": float(n_fin),
            "past_n": float(past), "pres_n": float(pres),
            "progressive_n": float(prog), "perfect_n": float(perf),
        })
    return out


# ---------------------------------------------------------------------------
# Family D part 1 - POS rates
# ---------------------------------------------------------------------------

def pos_rate_features(doc, include_counts: bool = True) -> Dict[str, float]:
    """POS rates per token and the content/function word ratio."""
    n = len(doc)
    counts = {"noun": 0, "verb": 0, "adj": 0, "adv": 0, "propn": 0}
    content = 0
    function = 0
    for tok in doc:
        p = tok.pos_
        if p == "NOUN":
            counts["noun"] += 1
        elif p == "VERB":
            counts["verb"] += 1
        elif p == "ADJ":
            counts["adj"] += 1
        elif p == "ADV":
            counts["adv"] += 1
        elif p == "PROPN":
            counts["propn"] += 1
        if p in _CONTENT_POS:
            content += 1
        else:
            function += 1

    out: Dict[str, float] = {f"{k}_rate": _safe_rate(v, n) for k, v in counts.items()}
    out["content_word_ratio"] = _safe_rate(content, function)
    if include_counts:
        out.update({f"{k}_n": float(v) for k, v in counts.items()})
        out["content_n"] = float(content)
        out["function_n"] = float(function)
    return out


# ---------------------------------------------------------------------------
# Family D part 2 - lexical diversity (report tier)
# ---------------------------------------------------------------------------

def _alpha_lemmas(doc) -> List[str]:
    """Lowercased lemmas restricted to alphabetic tokens (for TTR/MATTR)."""
    return [tok.lemma_.lower() for tok in doc if tok.is_alpha]


def diversity_features(doc, mattr_window: int = 50,
                       min_tokens: int = 20) -> Dict[str, float]:
    """
    Type-token ratio and moving-average TTR over alphabetic lemmas. Returns NaN
    below the min_tokens floor. MATTR falls back to plain TTR when the token
    count is below the window. Recommended at report level only.
    """
    lemmas = _alpha_lemmas(doc)
    n = len(lemmas)
    out: Dict[str, float] = {"ttr": np.nan, "mattr": np.nan}
    if n < min_tokens:
        return out

    out["ttr"] = len(set(lemmas)) / n

    if n < mattr_window:
        out["mattr"] = out["ttr"]
    else:
        ratios = []
        for i in range(n - mattr_window + 1):
            window = lemmas[i:i + mattr_window]
            ratios.append(len(set(window)) / mattr_window)
        out["mattr"] = float(np.mean(ratios))
    return out


# ---------------------------------------------------------------------------
# Family E - syntactic complexity
# ---------------------------------------------------------------------------

def _tree_depth(sent) -> int:
    """Depth of the dependency tree for a sentence (root -> deepest leaf)."""
    root = None
    for tok in sent:
        if tok.head == tok or tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return 0

    def depth(tok) -> int:
        children = list(tok.children)
        if not children:
            return 1
        return 1 + max(depth(c) for c in children)

    return depth(root)


def complexity_features(doc, min_tokens: int = 20,
                        min_sent_tokens: int = 3) -> Dict[str, float]:
    """
    Syntactic complexity averaged over sentences (skip sentences shorter than
    min_sent_tokens). Returns NaN below the min_tokens floor.
    """
    out: Dict[str, float] = {
        "mean_sent_len_tokens": np.nan,
        "mean_dep_distance": np.nan,
        "max_tree_depth": np.nan,
        "clause_density": np.nan,
        "subord_rate": np.nan,
    }
    if len(doc) < min_tokens:
        return out

    sents = [s for s in _iter_sents(doc) if len(s) >= min_sent_tokens]
    if not sents:
        return out

    sent_lens = [len(s) for s in sents]

    # Dependency distance is averaged over the same retained sentences as the
    # other complexity measures, so a flood of sub-min_sent_tokens fragments
    # (common in log-style narratives) can't skew it. Counting over the whole
    # doc here would be inconsistent with mean_sent_len_tokens / clause_density.
    dep_distances = []
    for s in sents:
        for tok in s:
            if tok.head != tok:  # skip root
                dep_distances.append(abs(tok.i - tok.head.i))

    depths = [_tree_depth(s) for s in sents]

    # clause_density / subord_rate are likewise scoped to retained sentences.
    n_fin = sum(len(_finite_verbs(s)) for s in sents)
    n_mark = sum(1 for s in sents for tok in s if tok.dep_ == "mark")

    out["mean_sent_len_tokens"] = float(np.mean(sent_lens))
    out["mean_dep_distance"] = float(np.mean(dep_distances)) if dep_distances else np.nan
    out["max_tree_depth"] = float(max(depths)) if depths else np.nan
    out["clause_density"] = _safe_rate(n_fin, len(sents))
    out["subord_rate"] = _safe_rate(n_mark, n_fin)
    return out


# ---------------------------------------------------------------------------
# Family F - Pennebaker composites (optional, approximate)
# ---------------------------------------------------------------------------

# Component rates and their signs for each Pennebaker-style composite. Positive
# components raise the score, negative lower it. These are POS-derived proxies
# for the LIWC categories Pennebaker uses; they are NOT LIWC and the exact
# weighting differs - document the deviation when reporting.
_ANALYTICAL_COMPONENTS = {
    "article_rate": +1.0, "prep_rate": +1.0,
    "pron_i_rate": -1.0, "pron_we_rate": -1.0, "pron_you_rate": -1.0,
    "pron_3rd_rate": -1.0, "aux_rate": -1.0,
}
_CLOUT_COMPONENTS = {
    "pron_we_rate": +1.0, "pron_you_rate": +1.0, "pron_i_rate": -1.0,
}


def composite_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Approximate Pennebaker composites from a single span's style rates.

    NOTE: this raw form simply sums/subtracts rates that live on different
    scales (article_rate ~0.10 dwarfs pron_i_rate ~0.02), so it is dominated by
    the largest-magnitude component rather than being a true standardized
    combination. It is kept only as a convenience for one-off scoring of a
    single span where no corpus is available. For analysis, prefer
    `standardize_composites`, which z-scores each component across the corpus
    first (what the design spec means by "standardized combination"). Either
    way, the multiverse should sweep over the components, not these summaries.

    analytical ~ favors articles + prepositions over pronouns + aux
    clout      ~ favors we + you over i
    """
    def g(k):
        return features.get(k, np.nan)

    analytical = np.nansum([w * g(k) for k, w in _ANALYTICAL_COMPONENTS.items()])
    clout = np.nansum([w * g(k) for k, w in _CLOUT_COMPONENTS.items()])
    return {"analytical_thinking": float(analytical), "clout": float(clout)}


def standardize_composites(
    df: pd.DataFrame,
    analytical_components: Optional[Dict[str, float]] = None,
    clout_components: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Corpus-level standardized Pennebaker composites.

    Takes a DataFrame of per-report features (one row per report, columns are
    the flat style-rate keys produced by `style_features`) and returns a copy
    with two added columns, `analytical_thinking` and `clout`, each computed by
    z-scoring every component rate across the corpus (population std) and then
    summing with +/-1 weights. z-scoring puts the components on a common scale
    before combination, so no single rate dominates by magnitude alone.

    Missing component columns are skipped (with their weight dropped); a
    component with zero variance contributes 0. Rows where every component is
    NaN yield NaN. This is the form to feed downstream models / reporting.
    """
    analytical_components = analytical_components or _ANALYTICAL_COMPONENTS
    clout_components = clout_components or _CLOUT_COMPONENTS
    out = df.copy()

    def _composite(components: Dict[str, float]) -> pd.Series:
        contribs = []
        any_present = pd.Series(False, index=out.index)
        for col, w in components.items():
            if col not in out.columns:
                continue
            vals = out[col].astype(float)
            mu = vals.mean(skipna=True)
            sd = vals.std(ddof=0, skipna=True)
            z = (vals - mu) / sd if sd and np.isfinite(sd) and sd > 0 else vals * 0.0
            contribs.append((w * z).fillna(0.0))
            any_present = any_present | vals.notna()
        if not contribs:
            return pd.Series(np.nan, index=out.index, dtype=float)
        total = sum(contribs)
        return total.where(any_present, np.nan)

    out["analytical_thinking"] = _composite(analytical_components)
    out["clout"] = _composite(clout_components)
    return out


# ---------------------------------------------------------------------------
# Top-level single-span extractor (Tier 1 families)
# ---------------------------------------------------------------------------

def _load_nlp(model: str):
    if spacy is None:
        raise ImportError("spaCy is required for linguistic features.")
    return spacy.load(model, disable=["ner"])


def extract_linguistic_features(
    doc_or_text: Union[str, "Doc", "Span"],
    config: LinguisticConfig = LinguisticConfig(),
    nlp: Optional["Language"] = None,
) -> Dict[str, float]:
    """
    Compute one row of Tier-1 linguistic features for a single segment/sentence.
    Diversity (Tier 2) is included here too when compute_diversity is set, but is
    only meaningful on report-length spans; prefer computing it at report level.
    Empty/degenerate input returns an all-NaN dict rather than raising.
    """
    if isinstance(doc_or_text, str):
        if nlp is None:
            nlp = _load_nlp(config.spacy_model)
        doc = nlp(doc_or_text)
    else:
        doc = doc_or_text

    feats: Dict[str, float] = {}
    n_tokens = len(doc)
    # A span with no alphabetic tokens (empty, whitespace, punctuation/IDs only)
    # has undefined linguistic features - return the all-NaN schema, not zeros.
    if n_tokens == 0 or not any(tok.is_alpha for tok in doc):
        return _nan_schema(config, nlp)

    if config.compute_passive:
        feats.update(passive_features(doc, config.include_counts))
    if config.compute_style:
        feats.update(style_features(doc, config.include_counts))
    if config.compute_tense:
        feats.update(tense_features(doc, config.compute_future, config.include_counts))
    if config.compute_pos_rates:
        feats.update(pos_rate_features(doc, config.include_counts))
    if config.compute_complexity:
        feats.update(complexity_features(
            doc, config.min_tokens_for_complexity, config.min_sent_tokens))
    if config.compute_diversity:
        feats.update(diversity_features(
            doc, config.mattr_window, config.min_tokens_for_diversity))
    if config.compute_composites and config.compute_style:
        feats.update(composite_features(feats))
    return feats


_SCHEMA_CACHE: Dict[tuple, List[str]] = {}


def _nan_schema(config: LinguisticConfig, nlp: Optional["Language"] = None) -> Dict[str, float]:
    """
    All-NaN feature dict matching the configured schema (for degenerate input).
    The key set is derived once per config from a known-good parse and cached,
    so empty segments in a batch don't reload the spaCy model.
    """
    key = tuple(sorted(config.__dict__.items()))
    keys = _SCHEMA_CACHE.get(key)
    if keys is None:
        if nlp is None:
            nlp = _load_nlp(config.spacy_model)
        probe = " ".join(["the inspector verified that valves were opened safely"] * 5)
        keys = list(extract_linguistic_features(nlp(probe), config, nlp).keys())
        _SCHEMA_CACHE[key] = keys
    return {k: np.nan for k in keys}


# ---------------------------------------------------------------------------
# Batch segment extractor
# ---------------------------------------------------------------------------

def extract_segment_features(
    segments: Sequence[str],
    config: LinguisticConfig = LinguisticConfig(),
    nlp: Optional["Language"] = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Batch-parse segments with nlp.pipe and return one row per segment. Adds
    'segment_index' and 'n_tokens'. Diversity is left to report-level
    aggregation, so it is disabled here regardless of config.
    """
    if nlp is None:
        nlp = _load_nlp(config.spacy_model)

    seg_cfg = LinguisticConfig(**{**config.__dict__, "compute_diversity": False})

    rows: List[Dict[str, float]] = []
    docs = nlp.pipe(list(segments), batch_size=batch_size)
    for i, doc in enumerate(docs):
        feats = extract_linguistic_features(doc, seg_cfg, nlp)
        feats["segment_index"] = i
        feats["n_tokens"] = len(doc)
        rows.append(feats)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation: segment-level -> report-level
# ---------------------------------------------------------------------------

def _resolve_weights(
    sf: pd.DataFrame,
    weights: Optional[Sequence[float]],
    config: AggregationConfig,
    segment_lengths: Optional[Sequence[float]],
) -> np.ndarray:
    """Resolve a per-segment weight vector for the chosen aggregation method."""
    n = len(sf)
    method = config.method

    if method == "mean":
        return np.ones(n, dtype=float)

    if method == "length_weighted":
        if segment_lengths is not None:
            w = np.asarray(segment_lengths, dtype=float)
        elif "n_tokens" in sf.columns:
            w = sf["n_tokens"].to_numpy(dtype=float)
        else:
            raise ValueError("length_weighted requires segment_lengths or an 'n_tokens' column.")
        return w

    # climate_* methods all require an aligned weights vector.
    if weights is None:
        raise ValueError(f"method '{method}' requires a weights vector.")
    if len(weights) != n:
        raise ValueError(
            f"weights length {len(weights)} != number of segments {n}; "
            "the caller must align weights to segment_features positionally."
        )
    w = np.asarray(weights, dtype=float)

    if method == "climate_weighted":
        return w
    if method == "climate_top_k":
        if config.k is None:
            raise ValueError("climate_top_k requires config.k.")
        keep = np.argsort(-w)[: config.k]
        mask = np.zeros(n, dtype=float)
        mask[keep] = 1.0
        return mask
    if method == "climate_threshold":
        if config.threshold is None:
            raise ValueError("climate_threshold requires config.threshold.")
        return (w >= config.threshold).astype(float)

    raise ValueError(f"unknown aggregation method: {method}")


def aggregate_segment_features(
    segment_features: pd.DataFrame,
    weights: Optional[Sequence[float]] = None,
    config: AggregationConfig = AggregationConfig(),
    segment_lengths: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Roll up Tier-1 segment features to one report-level row using a weighted
    nanmean per feature column. NaNs are dropped per-column before averaging
    (and their weights renormalized), so features below their stability floor on
    some segments don't poison the report value.

    For rate features ('*_rate', '*_per_sent', '*_per_clause') length_weighting
    recovers the report-level rate. Count columns ('*_n') are summed.
    """
    sf = segment_features.copy()
    meta_cols = {"segment_index", "n_tokens"}
    feat_cols = [c for c in sf.columns if c not in meta_cols]
    if len(sf) == 0:
        return {c: np.nan for c in feat_cols}

    base_w = _resolve_weights(sf, weights, config, segment_lengths)

    out: Dict[str, float] = {}
    for col in feat_cols:
        vals = sf[col].to_numpy(dtype=float)
        if col.endswith("_n"):
            out[col] = float(np.nansum(vals))
            continue
        finite = np.isfinite(vals)
        w = base_w.copy()
        w = np.where(finite, w, 0.0)
        wsum = w.sum()
        if wsum <= 0 or not finite.any():
            out[col] = np.nan
            continue
        if config.normalize_weights:
            w = w / wsum
            out[col] = float(np.nansum(np.where(finite, vals, 0.0) * w))
        else:
            out[col] = float(np.nansum(np.where(finite, vals, 0.0) * w) / wsum)
    return out


# ---------------------------------------------------------------------------
# Stateful wrapper (parallels SafetyFeatureExtractor)
# ---------------------------------------------------------------------------

class LinguisticFeatureExtractor:
    """
    Holds a loaded spaCy pipeline so it isn't reloaded per call.

    extract(text)              -> Tier-1 + Tier-2 features for one span/report.
    extract_segments(segments) -> per-segment Tier-1 DataFrame.
    extract_report(text, ...)  -> report-level row: Tier-1 aggregated over
                                  segments + Tier-2 diversity computed on the
                                  whole report.
    """

    def __init__(self, config: LinguisticConfig = LinguisticConfig()):
        self.config = config
        self.nlp = _load_nlp(config.spacy_model)

    def extract(self, text: str) -> Dict[str, float]:
        return extract_linguistic_features(text, self.config, self.nlp)

    def extract_segments(self, segments: Sequence[str]) -> pd.DataFrame:
        return extract_segment_features(segments, self.config, self.nlp)

    def extract_report(
        self,
        text: str,
        segmenter=None,
        agg_config: AggregationConfig = AggregationConfig(),
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """
        Segment -> per-segment Tier-1 features -> aggregate to report, then add
        report-level Tier-2 diversity computed on the full text.

        segmenter: optional object with a .segment(text) -> List[str] method
        (e.g. SentenceSegmenter from the safety toolkit). When provided, this is
        the climate-aligned mode and `weights` must align positionally with the
        segments it returns. When None, the report is split into spaCy sentences
        (independent mode: all sentences, no climate weighting).
        """
        if segmenter is not None:
            segments = list(segmenter.segment(text or ""))
        else:
            doc = self.nlp(text or "")
            segments = [s.text for s in _iter_sents(doc)]

        if weights is not None and len(weights) != len(segments):
            raise ValueError(
                f"weights length {len(weights)} != segment count {len(segments)}; "
                "align weights to the segmenter output (drop weights for dropped segments)."
            )

        sf = self.extract_segments(segments)
        report = aggregate_segment_features(sf, weights, agg_config)

        # Tier 2: diversity on the whole report (not climate-weightable).
        if self.config.compute_diversity:
            full_doc = self.nlp(text or "")
            report.update(diversity_features(
                full_doc, self.config.mattr_window, self.config.min_tokens_for_diversity))
        report["n_segments"] = float(len(segments))
        return report


# ---------------------------------------------------------------------------
# Namespacing for downstream (R) consumption
# ---------------------------------------------------------------------------

_FAMILY_OF = {
    "passive": "passive",
    "pron": "style", "article": "style", "prep": "style", "cconj": "style",
    "sconj": "style", "aux": "style", "modal": "style", "neg": "style", "quant": "style",
    "past": "tense", "pres": "tense", "future": "tense", "progressive": "tense",
    "perfect": "tense", "fin_verb": "tense",
    "noun": "pos", "verb": "pos", "adj": "pos", "adv": "pos", "propn": "pos",
    "content": "pos", "function": "pos",
    "ttr": "diversity", "mattr": "diversity",
    "mean_sent": "complexity", "mean_dep": "complexity", "max_tree": "complexity",
    "clause": "complexity", "subord": "complexity",
    "analytical": "composite", "clout": "composite",
}


def namespace_features(feats: Dict[str, float], prefix: str = "ling") -> Dict[str, float]:
    """
    Rename flat feature keys to '<prefix>__<family>__<feature>' for clean R-side
    joins (parallels temp__/ews__). Unknown keys fall back to '<prefix>__<key>'.
    """
    out: Dict[str, float] = {}
    for k, v in feats.items():
        family = None
        for stem, fam in _FAMILY_OF.items():
            if k.startswith(stem):
                family = fam
                break
        out[f"{prefix}__{family}__{k}" if family else f"{prefix}__{k}"] = v
    return out
