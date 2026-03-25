import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
     compute_bertscore,
    compute_sentence_transformer_similarity,
    compute_wmd_full_text,
    extract_keywords_single_model,
    extract_keywords_combined_paper_mode,
    extract_keywords_yake,
    compute_exact_match,
    compute_partial_match,
    compute_semantic_pairwise_similarity,
    compute_set_level_embedding_similarity,
    compute_wmd_keywords,
)

st.set_page_config(page_title="LLM Evaluator", layout="wide")




# -----------------------------
# Helper functions
# -----------------------------
def plot_metric_bars(metrics: dict, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(metrics.keys()), list(metrics.values()))
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


def render_metric_cards(metrics: dict):
    cols = st.columns(len(metrics))
    for idx, (name, value) in enumerate(metrics.items()):
        with cols[idx]:
            if isinstance(value, float):
                st.metric(name, f"{value:.4f}")
            else:
                st.metric(name, value)


def build_keyword_match_table(reference_keywords: list[str], generated_keywords: list[str]) -> pd.DataFrame:
    rows = []
    for ref_kw in reference_keywords:
        match = ""
        match_type = "Missing"

        if ref_kw in generated_keywords:
            match = ref_kw
            match_type = "Exact match"
        else:
            partial = next((g for g in generated_keywords if ref_kw in g or g in ref_kw), None)
            if partial:
                match = partial
                match_type = "Partial match"

        rows.append(
            {
                "Reference keyword": ref_kw,
                "Generated keyword": match,
                "Match type": match_type,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("LLM Evaluator")
mode = st.sidebar.radio(
    "Choose evaluation mode",
    ["Full-text evaluation", "Keyword evaluation"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.write(
    "Paste a reference text and a generated text, select one or more metrics, then run the evaluation."
)


# -----------------------------
# Main layout
# -----------------------------
st.title("LLM Output Evaluation Tool")
st.write(
    "Compare a generated text against a reference text using the evaluation approaches from your research."
)

left_col, right_col = st.columns([1.05, 1])

with left_col:
    st.subheader("Input texts")
    reference_text = st.text_area(
        "Reference text",
        height=220,
        placeholder="Paste the reference/expert answer here...",
    )
    generated_text = st.text_area(
        "Generated text",
        height=220,
        placeholder="Paste the generated/LLM answer here...",
    )

with right_col:
    keyword_extraction_mode = None
    keyword_model_name = None

    if mode == "Keyword evaluation":
        st.subheader("Keyword extraction options")

        keyword_extraction_mode = st.radio(
            "Keyword extraction strategy",
            [
                "Single-model keyword extraction",
                "Combined paper mode (General + Legal + Medical)",
                "YAKE keyword extraction",
            ],
        )

        if keyword_extraction_mode == "Single-model keyword extraction":
            keyword_model_name = st.selectbox(
                "Keyword extraction model",
                ["General", "Legal", "Medical"],
                index=1,
            )

        st.markdown("---")

    st.subheader("Evaluation settings")

    if mode == "Full-text evaluation":
        selected_metrics = st.multiselect(
            "Select full-text metrics",
            [
                "BERTScore (Precision, Recall, F1)",
                "Sentence-transformer semantic similarity",
                "Word Mover's Distance",
            ],
            default=["BERTScore (Precision, Recall, F1)"],
        )

    elif mode == "Keyword evaluation":
        selected_metrics = st.multiselect(
            "Select keyword metrics",
            [
                "Exact match",
                "Partial match",
                "Semantic pairwise similarity",
                "Set-level embedding similarity",
                "Word Mover's Distance",
            ],
            default=["Exact match"],
        )

    run_evaluation = st.button("Evaluate", type="primary", use_container_width=True)

# -----------------------------
# Results
# -----------------------------
st.markdown("---")
st.subheader("Results")

if run_evaluation:
    if not reference_text.strip() or not generated_text.strip():
        st.error("Please provide both the reference text and the generated text.")
    elif not selected_metrics:
        st.error("Please select at least one metric.")
    else:
        if mode == "Full-text evaluation":
            results = {}

            if "BERTScore (Precision, Recall, F1)" in selected_metrics:
                with st.spinner("Computing BERTScore..."):
                    bert = compute_bertscore(reference_text, generated_text)
                results["BERTScore Precision"] = bert["Precision"]
                results["BERTScore Recall"] = bert["Recall"]
                results["BERTScore F1"] = bert["F1"]

            if "Sentence-transformer semantic similarity" in selected_metrics:
                with st.spinner("Computing sentence-transformer semantic similarity..."):
                    results["Sentence-transformer semantic similarity"] = compute_sentence_transformer_similarity(
                        reference_text, generated_text
                    )

            if "Word Mover's Distance" in selected_metrics:
                with st.spinner("Computing Word Mover's Distance..."):
                    results["Word Mover's Distance"] = compute_wmd_full_text(reference_text, generated_text)

            render_metric_cards(results)
            st.markdown("### Visualization")
            plot_metric_bars(results, "Full-text evaluation metrics")

            st.markdown("### Evaluation summary")
            st.info(
                "This is a placeholder summary. Once your real metric code is connected, this section can generate a research-based interpretation automatically."
            )

        else:
            st.markdown("### Extracted keywords")

            if keyword_extraction_mode == "Single-model keyword extraction":
                with st.spinner(f"Extracting keywords using the {keyword_model_name} model..."):
                    ref_keywords = extract_keywords_single_model(reference_text, keyword_model_name)
                    gen_keywords = extract_keywords_single_model(generated_text, keyword_model_name)

                kw_col1, kw_col2 = st.columns(2)
                with kw_col1:
                    st.write("**Reference keywords**")
                    st.write(ref_keywords)
                with kw_col2:
                    st.write("**Generated keywords**")
                    st.write(gen_keywords)

            elif keyword_extraction_mode == "Combined paper mode (General + Legal + Medical)":
                with st.spinner("Extracting General, Legal, and Medical keywords..."):
                    ref_kw_data = extract_keywords_combined_paper_mode(reference_text)
                    gen_kw_data = extract_keywords_combined_paper_mode(generated_text)

                ref_keywords = ref_kw_data["Combined"]
                gen_keywords = gen_kw_data["Combined"]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Reference - General**")
                    st.write(ref_kw_data["General"])
                    st.write("**Reference - Legal**")
                    st.write(ref_kw_data["Legal"])
                    st.write("**Reference - Medical**")
                    st.write(ref_kw_data["Medical"])
                    st.write("**Reference - Combined**")
                    st.write(ref_kw_data["Combined"])

                with col2:
                    st.write("**Generated - General**")
                    st.write(gen_kw_data["General"])
                    st.write("**Generated - Legal**")
                    st.write(gen_kw_data["Legal"])
                    st.write("**Generated - Medical**")
                    st.write(gen_kw_data["Medical"])
                    st.write("**Generated - Combined**")
                    st.write(gen_kw_data["Combined"])

            elif keyword_extraction_mode == "YAKE keyword extraction":
                with st.spinner("Extracting YAKE keywords..."):
                     ref_keywords = extract_keywords_yake(reference_text)
                     gen_keywords = extract_keywords_yake(generated_text)

                kw_col1, kw_col2 = st.columns(2)
                with kw_col1:
                     st.write("**Reference YAKE keywords**")
                     st.write(ref_keywords)
                with kw_col2:
                     st.write("**Generated YAKE keywords**")
                     st.write(gen_keywords)

            results = {}
            if "Exact match" in selected_metrics:
                results["Exact match"] = compute_exact_match(ref_keywords, gen_keywords)
            if "Partial match" in selected_metrics:
                results["Partial match"] = compute_partial_match(ref_keywords, gen_keywords)
            if "Semantic pairwise similarity" in selected_metrics:
                results["Semantic pairwise similarity"] = compute_semantic_pairwise_similarity(ref_keywords, gen_keywords)
            if "Set-level embedding similarity" in selected_metrics:
                results["Set-level embedding similarity"] = compute_set_level_embedding_similarity(ref_keywords, gen_keywords)
            if "Word Mover's Distance" in selected_metrics:
                results["Keyword Word Mover's Distance"] = compute_wmd_keywords(ref_keywords, gen_keywords)

            st.markdown("### Metric summary")
            render_metric_cards(results)

            st.markdown("### Visualization")
            plot_metric_bars(results, "Keyword evaluation metrics")

            st.markdown("### Keyword comparison table")
            match_df = build_keyword_match_table(ref_keywords, gen_keywords)
            st.dataframe(match_df, use_container_width=True)

            st.markdown("### Evaluation summary")
            st.info(
                "This is a placeholder summary. Once your real keyword extraction and matching code is connected, this section can describe missing and matched concepts more precisely."
            )
else:
    st.caption("Results will appear here after you run an evaluation.")


