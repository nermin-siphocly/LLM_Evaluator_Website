import re
import streamlit as st

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


import yake




# =========================
# BERTScore
# =========================
import streamlit as st


@st.cache_resource
def get_bertscore_import():
    from bert_score import score
    return score


def compute_bertscore(reference_text: str, generated_text: str) -> dict:
    if not reference_text.strip() or not generated_text.strip():
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    score_fn = get_bertscore_import()
    P, R, F1 = score_fn([generated_text], [reference_text], lang="en", verbose=False)

    return {
        "Precision": float(P.mean().item()),
        "Recall": float(R.mean().item()),
        "F1": float(F1.mean().item()),
    }

# =========================
# Full-text semantic similarity
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


def compute_wmd_full_text(reference_text, generated_text):
    if not reference_text.strip() or not generated_text.strip():
        return float("inf")

    # Load model only when needed (cached)
    word_vectors = get_wmd_model()

    t1 = preprocess_for_wmd(reference_text, word_vectors)
    t2 = preprocess_for_wmd(generated_text, word_vectors)

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

    general_keywords = extract_keywords_single_model(text, "General", top_n, nr_candidates)
    legal_keywords = extract_keywords_single_model(text, "Legal", top_n, nr_candidates)
    medical_keywords = extract_keywords_single_model(text, "Medical", top_n, nr_candidates)

    combined_keywords = list(set(general_keywords + legal_keywords + medical_keywords))

    return {
        "General": general_keywords,
        "Legal": legal_keywords,
        "Medical": medical_keywords,
        "Combined": combined_keywords,
    }

#-----------------------------------------------------

# =========================
# Keyword comparison models
# =========================
@st.cache_resource
def get_keyword_semantic_comparison_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# Keyword comparison metrics
# =========================
def compute_exact_match(reference_keywords: list[str], generated_keywords: list[str]) -> float:
    """
    
    exact_count = sum(1 for expr in set2 if expr in set1)
    exact_score = exact_count / len(set2) if set2 else 0
    """
    set2 = reference_keywords
    set1 = generated_keywords

    if not set2:
        return 0.0

    exact_count = sum(1 for expr in set2 if expr in set1)
    return exact_count / len(set2)


def compute_semantic_pairwise_similarity(
    reference_keywords: list[str],
    generated_keywords: list[str],
    threshold: float = 0.8,
) -> float:
    """
    
    - encode generated and reference keyword sets
    - for each reference keyword, check whether any generated keyword
      reaches cosine similarity >= threshold
    - score = matched_reference_keywords / len(reference_keywords)
    """
    set2 = reference_keywords
    set1 = generated_keywords

    if not set2 or not set1:
        return 0.0

    semantic_model = get_keyword_semantic_comparison_model()

    embeddings1 = semantic_model.encode(set1)
    embeddings2 = semantic_model.encode(set2)

    sem_count = 0
    for i, expr2 in enumerate(set2):
        for j, expr1 in enumerate(set1):
            similarity = cosine_similarity([embeddings2[i]], [embeddings1[j]])[0][0]
            if similarity >= threshold:
                sem_count += 1
                break

    return sem_count / len(set2)


def compute_set_level_embedding_similarity(
    reference_keywords: list[str],
    generated_keywords: list[str],
) -> float:
    """
   
    - encode each keyword/keyphrase
    - average embeddings within each set
    - cosine similarity between average embeddings
    """
    set2 = reference_keywords
    set1 = generated_keywords

    if not set2 or not set1:
        return 0.0

    semantic_model = get_keyword_semantic_comparison_model()

    embeddings1 = semantic_model.encode(set1)
    embeddings2 = semantic_model.encode(set2)

    avg_embedding1 = np.mean(embeddings1, axis=0)
    avg_embedding2 = np.mean(embeddings2, axis=0)

    return float(cosine_similarity([avg_embedding1], [avg_embedding2])[0][0])


def compute_partial_match(
    reference_keywords: list[str],
    generated_keywords: list[str],
    word_overlap_threshold: int = 2,
) -> float:
    """
   
    - for each reference keyword phrase
    - compare word overlap with each generated keyword phrase
    - count as matched if overlap >= threshold
    """
    set2 = reference_keywords
    set1 = generated_keywords

    if not set2:
        return 0.0

    partial_count = 0
    for expr2 in set2:
        words2 = set(expr2.split())
        for expr1 in set1:
            words1 = set(expr1.split())
            if len(words1.intersection(words2)) >= word_overlap_threshold:
                partial_count += 1
                break

    return partial_count / len(set2)


def compute_wmd_keywords(reference_keywords: list[str], generated_keywords: list[str]) -> float:
    """
    
    - flatten keyword phrases into word lists
    - compute WMD over those words
    """
    set2 = reference_keywords
    set1 = generated_keywords

    if not set2 or not set1:
        return float("inf")

    wmd_model = get_wmd_model()

    words1 = [word for phrase in set1 for word in phrase.lower().split()]
    words2 = [word for phrase in set2 for word in phrase.lower().split()]

    # Keep only tokens that exsist in the embedding vocab
    words1 = [word for word in words1 if word in wmd_model]
    words2 = [word for word in words2 if word in wmd_model]

    if not words1 or not words2:
        return float("inf")

    try:
        distance = wmd_model.wmdistance(words1, words2)
    except Exception:
        distance = float("inf")

    return float(distance)

#-----------------------------------------------------

# =========================
# YAKE keyword extraction
# =========================
@st.cache_resource
def get_yake_extractor():
    # Faithful to the notebook:
    # lan="en", n=3, top=10
    return yake.KeywordExtractor(lan="en", n=3, top=10)


def extract_keywords_yake(text: str) -> list[str]:
    """
    Faithful adaptation of the YAKE extraction notebook:
    - clean text
    - extract up to 10 keywords/keyphrases
    - keep only the keyword strings
    """
    if not text.strip():
        return []

    cleaned_text = clean_text_for_keywords(text)
    kw_extractor = get_yake_extractor()

    try:
        keywords_with_scores = kw_extractor.extract_keywords(cleaned_text)
        keywords_only = [kw for kw, score in keywords_with_scores]
        return keywords_only
    except Exception:
        return []
