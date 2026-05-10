import re
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import yake


# =========================
# BERTScore
# PATCH 1: Cache BERTScorer object instead of just the import.
# The old approach re-loaded the underlying BERT model on every call.
# BERTScorer keeps the model in memory after the first load.
# =========================

@st.cache_resource
def get_bertscore_scorer():
    from bert_score import BERTScorer
    return BERTScorer(lang="en", rescale_with_baseline=False, verbose=False)


def compute_bertscore(reference_text: str, generated_text: str) -> dict:
    if not reference_text.strip() or not generated_text.strip():
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    scorer = get_bertscore_scorer()
    P, R, F1 = scorer.score([generated_text], [reference_text])

    return {
        "Precision": float(P.mean()),
        "Recall": float(R.mean()),
        "F1": float(F1.mean()),
    }


# =========================
# Full-text semantic similarity
# (unchanged — already efficient via cached model)
# =========================

@st.cache_resource
def get_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_sentence_transformer_similarity(reference_text: str, generated_text: str) -> float:
    if not reference_text.strip() or not generated_text.strip():
        return 0.0

    from sentence_transformers import util

    model = get_semantic_model()

    reference = re.sub(r"[^a-zA-Z0-9.;,!?/\|@&#$\-_: ]", "", reference_text)
    generated = re.sub(r"[^a-zA-Z0-9.;,!?/\|@&#$\-_: ]", "", generated_text)

    emb1 = model.encode(reference, convert_to_tensor=True)
    emb2 = model.encode(generated, convert_to_tensor=True)

    return float(util.pytorch_cos_sim(emb1, emb2).item())


# =========================
# Cached WMD model
# =========================

@st.cache_resource
def get_wmd_model():
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-100")


def preprocess_for_wmd(text: str, word_vectors):
    from gensim.utils import simple_preprocess

    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = simple_preprocess(cleaned)
    tokens = [t for t in tokens if t in word_vectors]
    return tokens


# PATCH 2: Truncate tokens before WMD on full text.
# WMD complexity is O(n³ log n). A 500-token document can take
# 30–60× longer than a 100-token one with negligible accuracy gain.
MAX_WMD_TOKENS = 100


def compute_wmd_full_text(reference_text, generated_text):
    if not reference_text.strip() or not generated_text.strip():
        return float("inf")

    word_vectors = get_wmd_model()

    t1 = preprocess_for_wmd(reference_text, word_vectors)[:MAX_WMD_TOKENS]
    t2 = preprocess_for_wmd(generated_text, word_vectors)[:MAX_WMD_TOKENS]

    if not t1 or not t2:
        return float("inf")

    dist = word_vectors.wmdistance(t1, t2)
    avg_len = (len(t1) + len(t2)) / 2

    return dist / avg_len if avg_len > 0 else float("inf")


# =========================
# Keyword extraction
# =========================

@st.cache_resource
def get_keyword_model(model_name: str):
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    if model_name == "General":
        st_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    elif model_name == "Legal":
        st_model = SentenceTransformer("nlpaueb/bert-base-uncased-eurlex")
    elif model_name == "Medical":
        st_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    else:
        raise ValueError(f"Unknown keyword model: {model_name}")

    return KeyBERT(model=st_model)


def clean_text_for_keywords(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.;,!?/\|@&#$ \n]", "", text)


@st.cache_data
def extract_keywords_single_model(
    text: str,
    model_name: str,
    top_n: int = 10,
    nr_candidates: int = 20,
) -> list[str]:
    if not text.strip():
        return []

    cleaned_text = clean_text_for_keywords(text)
    kw_model = get_keyword_model(model_name)

    keywords_with_scores = kw_model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=nr_candidates,
        top_n=top_n,
    )

    return [keyword for keyword, score in keywords_with_scores]


# PATCH 5: Parallelize Combined paper mode.
# The 3 models are independent — running them in parallel cuts wall-clock
# time from ~3× single-model time down to ~1× single-model time.
@st.cache_data
def extract_keywords_combined_paper_mode(
    text: str,
    top_n: int = 10,
    nr_candidates: int = 20,
) -> dict:
    if not text.strip():
        return {
            "General": [],
            "Legal": [],
            "Medical": [],
            "Combined": [],
        }

    model_names = ["General", "Legal", "Medical"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            name: executor.submit(
                extract_keywords_single_model, text, name, top_n, nr_candidates
            )
            for name in model_names
        }
        extracted = {name: futures[name].result() for name in model_names}

    extracted["Combined"] = list(set(
        extracted["General"] + extracted["Legal"] + extracted["Medical"]
    ))
    return extracted


# =========================
# YAKE keyword extraction
# =========================

@st.cache_resource
def get_yake_extractor():
    return yake.KeywordExtractor(lan="en", n=3, top=10)


def extract_keywords_yake(text: str) -> list[str]:
    if not text.strip():
        return []

    cleaned_text = clean_text_for_keywords(text)
    kw_extractor = get_yake_extractor()

    try:
        keywords_with_scores = kw_extractor.extract_keywords(cleaned_text)
        return [kw for kw, score in keywords_with_scores]
    except Exception:
        return []


# =========================
# Keyword comparison: shared embedding model
# =========================

@st.cache_resource
def get_keyword_semantic_comparison_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# Keyword comparison metrics
# =========================

def compute_exact_match(reference_keywords: list[str], generated_keywords: list[str]) -> float:
    if not reference_keywords:
        return 0.0
    exact_count = sum(1 for kw in reference_keywords if kw in generated_keywords)
    return exact_count / len(reference_keywords)


def compute_partial_match(
    reference_keywords: list[str],
    generated_keywords: list[str],
    word_overlap_threshold: int = 2,
) -> float:
    if not reference_keywords:
        return 0.0

    partial_count = 0
    for ref_kw in reference_keywords:
        words_ref = set(ref_kw.split())
        for gen_kw in generated_keywords:
            words_gen = set(gen_kw.split())
            if len(words_ref.intersection(words_gen)) >= word_overlap_threshold:
                partial_count += 1
                break

    return partial_count / len(reference_keywords)


# PATCH 3 & 4: Vectorized cosine similarity + single encoding pass.
# The old nested loop called cosine_similarity up to n×m times separately.
# This function encodes once and returns BOTH pairwise and set-level scores,
# so callers that need both metrics avoid encoding the same keywords twice.
def compute_embedding_metrics_combined(
    reference_keywords: list[str],
    generated_keywords: list[str],
    threshold: float = 0.8,
) -> dict:
    """
    Encode reference and generated keywords once, then compute:
      - semantic_pairwise: fraction of reference keywords that have at least
        one generated keyword with cosine similarity >= threshold.
      - set_level: cosine similarity between the mean embeddings of each set.

    Returns dict with keys 'semantic_pairwise' and 'set_level'.
    """
    if not reference_keywords or not generated_keywords:
        return {"semantic_pairwise": 0.0, "set_level": 0.0}

    model = get_keyword_semantic_comparison_model()
    emb_gen = model.encode(generated_keywords)   # shape: (n_gen, dim)
    emb_ref = model.encode(reference_keywords)   # shape: (n_ref, dim)

    # Full similarity matrix in one vectorized BLAS call — shape: (n_ref, n_gen)
    sim_matrix = cosine_similarity(emb_ref, emb_gen)

    # Pairwise: a reference keyword is matched if ANY generated keyword >= threshold
    sem_count = int((sim_matrix.max(axis=1) >= threshold).sum())
    pairwise = sem_count / len(reference_keywords)

    # Set-level: cosine similarity between mean embeddings
    set_level = float(cosine_similarity(
        [np.mean(emb_gen, axis=0)],
        [np.mean(emb_ref, axis=0)]
    )[0][0])

    return {"semantic_pairwise": pairwise, "set_level": set_level}


# Convenience wrappers so existing call sites in LLM_Eval.py keep working
# (though it is more efficient to call compute_embedding_metrics_combined once
# and unpack both values when both metrics are selected — see LLM_Eval.py patch).
def compute_semantic_pairwise_similarity(
    reference_keywords: list[str],
    generated_keywords: list[str],
    threshold: float = 0.8,
) -> float:
    return compute_embedding_metrics_combined(
        reference_keywords, generated_keywords, threshold
    )["semantic_pairwise"]


def compute_set_level_embedding_similarity(
    reference_keywords: list[str],
    generated_keywords: list[str],
) -> float:
    return compute_embedding_metrics_combined(
        reference_keywords, generated_keywords
    )["set_level"]


def compute_wmd_keywords(reference_keywords: list[str], generated_keywords: list[str]) -> float:
    if not reference_keywords or not generated_keywords:
        return float("inf")

    wmd_model = get_wmd_model()

    words_gen = [word for phrase in generated_keywords for word in phrase.lower().split()]
    words_ref = [word for phrase in reference_keywords  for word in phrase.lower().split()]

    words_gen = [w for w in words_gen if w in wmd_model]
    words_ref = [w for w in words_ref if w in wmd_model]

    if not words_gen or not words_ref:
        return float("inf")

    try:
        return float(wmd_model.wmdistance(words_gen, words_ref))
    except Exception:
        return float("inf")
