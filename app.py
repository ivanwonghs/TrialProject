import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Optional, List, Tuple
import pandas as pd
import io
import csv
import time

st.set_page_config(page_title="Multilingual Comment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Hard-coded model IDs (used internally)
_TRANSLATE_MODEL_ID = "Qwen/Qwen3-0.6B"
_SENTIMENT_MODEL_ID = "ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews"

# Cached pipeline dictionaries (keyed by model id)
_SENTIMENT_PIPELINES: dict = {}
_TRANSLATE_PIPELINES: dict = {}
_TRANSLATE_TOKENIZERS: dict = {}

@st.cache_resource
def get_sentiment_pipeline_cached(model_name: str):
    if model_name not in _SENTIMENT_PIPELINES:
        _SENTIMENT_PIPELINES[model_name] = pipeline(model=model_name)
    return _SENTIMENT_PIPELINES[model_name]

@st.cache_resource
def get_translate_pipeline_and_tokenizer_cached(model_name: str = _TRANSLATE_MODEL_ID):
    if model_name not in _TRANSLATE_PIPELINES:
        _TRANSLATE_PIPELINES[model_name] = pipeline("text-generation", model=model_name)
    if model_name not in _TRANSLATE_TOKENIZERS:
        _TRANSLATE_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINES[model_name], _TRANSLATE_TOKENIZERS[model_name]

# --- Core inference helpers --------------------------------------------------
def run_sentiment_single(text: str) -> Tuple[str, float]:
    """Return (label, confidence) for a single text input."""
    pipeline_obj = get_sentiment_pipeline_cached(_SENTIMENT_MODEL_ID)
    result = pipeline_obj(text)
    label = result[0].get("label", "UNKNOWN")
    score = result[0].get("score", 0.0)
    return label, score

def run_translate_single(text: str) -> str:
    """Return English meaning/translation for a single text input."""
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer_cached(_TRANSLATE_MODEL_ID)
    content_to_translate = "Translate the following into English: '" + text + "'"
    messages = [{"role": "user", "content": content_to_translate}]
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        text_input = content_to_translate

    outputs = translate_pipeline(text_input, max_new_tokens=1024)
    generated_text_full = outputs[0].get('generated_text', "")

    marker_end_think = "</think>\n\n"
    start_of_response_idx = generated_text_full.rfind(marker_end_think)
    if start_of_response_idx != -1:
        raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
    else:
        raw_response = generated_text_full

    return raw_response.strip().strip('"')

# --- Batch processing -------------------------------------------------------
def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Accept a CSV or TXT uploaded file and return a DataFrame with a single column 'comment'."""
    if uploaded_file is None:
        return pd.DataFrame(columns=["comment"])

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            # try to find the most likely text column
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if not text_cols:
                # fall back to reading whole file as single column
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=None, names=["comment"])
            else:
                # pick first text column
                df = df[[text_cols[0]]].rename(columns={text_cols[0]: "comment"})
        except Exception:
            # fallback: try reading with python engine
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, engine="python", encoding="utf-8", error_bad_lines=False)
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if text_cols:
                df = df[[text_cols[0]]].rename(columns={text_cols[0]: "comment"})
            else:
                df = pd.DataFrame(columns=["comment"])
    elif name.endswith(".txt"):
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        df = pd.DataFrame({"comment": lines})
    else:
        # try to parse as CSV anyway
        try:
            df = pd.read_csv(uploaded_file)
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if text_cols:
                df = df[[text_cols[0]]].rename(columns={text_cols[0]: "comment"})
            else:
                df = pd.DataFrame(columns=["comment"])
        except Exception:
            df = pd.DataFrame(columns=["comment"])

    # Ensure column exists
    if "comment" not in df.columns:
        df = df.rename(columns={df.columns[0]: "comment"}) if len(df.columns) > 0 else pd.DataFrame(columns=["comment"])

    # Drop empty comments
    df["comment"] = df["comment"].astype(str).fillna("").str.strip()
    df = df[df["comment"] != ""].reset_index(drop=True)
    return df

def process_batch(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Process rows [start_idx:end_idx) and return a DataFrame with results."""
    results = []
    total = max(0, end_idx - start_idx)
    for i, comment in enumerate(df["comment"].iloc[start_idx:end_idx].tolist(), start=0):
        try:
            label, score = run_sentiment_single(comment)
        except Exception:
            label, score = "ERROR", 0.0
        try:
            meaning = run_translate_single(comment)
        except Exception:
            meaning = ""
        results.append({
            "original_comment": comment,
            "sentiment_label": label,
            "sentiment_confidence": score,
            "meaning_in_english": meaning
        })
    return pd.DataFrame(results)

# --- UI --------------------------------------------------------------------
def main():
    st.markdown("## Multilingual Social Media Product Comment Analyzer\n")

    # Page layout CSS
    st.markdown(
        """
        <style>
        .layout {
            display: flex;
            min-height: 100vh;
            gap: 24px;
            align-items: flex-start;
            padding-top: 12px;
        }
        .left {
            width: 360px;
            padding: 16px;
            box-sizing: border-box;
            border-right: 1px solid rgba(0,0,0,0.08);
        }
        .right {
            flex: 1 1 auto;
            padding: 16px;
            box-sizing: border-box;
        }
        @media (max-width: 900px) {
            .layout { flex-direction: column; }
            .left { width: auto; border-right: none; border-bottom: 1px solid rgba(0,0,0,0.08); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="layout">', unsafe_allow_html=True)
    left_col, right_col = st.columns([0.28, 0.72])

    # Left column: About + Supported languages + Short instructions for Batch Mode
    with left_col:
        st.markdown('<div class="left">', unsafe_allow_html=True)
        st.markdown("### About this analyzer")
        st.markdown(
            """
            This application helps teams quickly assess short social-media or product comments written in many languages.

            How it works:
            - You can analyze a single comment (right side) or use Batch Mode to process many comments from a file.
            - For each comment the app returns a sentiment label (with confidence) and an English rendering/paraphrase.
            """
        )
        st.markdown("#### Batch Mode — quick guide")
        st.markdown(
            """
            - Upload a CSV (one text column) or a TXT file (one comment per line).
            - Preview the first rows, choose which rows to process, then click "Run batch".
            - When finished you can download the results as a CSV.
            """
        )
        st.write("")
        st.markdown("#### Supported languages")
        st.markdown(
            """
            - English  - Arabic  - German  - Spanish
            - French   - Japanese - Chinese - Indonesian
            - Hindi    - Italian  - Malay    - Portuguese
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Right column: Single comment mode + Batch mode
    with right_col:
        st.markdown('<div class="right">', unsafe_allow_html=True)

        st.markdown("### Single comment")
        st.markdown("Enter a comment below and click Analyze. The app will return a sentiment label with confidence and an English translation/meaning.")
        user_input = st.text_input("Please input the comment you want to analyse:", key="single_input")

        status_placeholder = st.empty()
        sentiment_placeholder = st.empty()
        translate_placeholder = st.empty()

        if st.button("Analyze", key="analyze_single"):
            if not user_input:
                st.warning("Please enter a comment to analyze.")
            else:
                with st.spinner("Analyzing comment — this may take a while..."):
                    status_placeholder.info("Running analysis. Please wait...")
                    try:
                        label, score = run_sentiment_single(user_input)
                        sentiment_placeholder.markdown(f"**Sentiment:** {label} ({score:.2%} confidence)")
                    except Exception:
                        sentiment_placeholder.error("Sentiment analysis failed. Please try again later.")
                    try:
                        meaning = run_translate_single(user_input)
                        translate_placeholder.markdown(f"**Meaning in English:** {meaning}")
                    except Exception:
                        translate_placeholder.error("Translation failed. Please try again later.")
                    status_placeholder.success("Analysis complete.")

        st.markdown("---")
        st.markdown("### Batch mode")
        st.markdown("Upload a CSV (one text column) or a TXT file (one comment per line). Preview, select rows, then run a batch analysis.")

        uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"], accept_multiple_files=False, key="batch_upload")

        df_preview = pd.DataFrame(columns=["comment"])
        if uploaded_file is not None:
            try:
                df_preview = parse_uploaded_file(uploaded_file)
            except Exception:
                st.error("Failed to read the uploaded file. Please check the file format.")
                df_preview = pd.DataFrame(columns=["comment"])

        if not df_preview.empty:
            st.markdown("#### Preview (first 10 rows)")
            st.dataframe(df_preview.head(10), use_container_width=True)

            total_rows = len(df_preview)
            st.markdown(f"Detected {total_rows} comments.")

            # Row selection
            col_a, col_b = st.columns(2)
            with col_a:
                start_idx = st.number_input("Start row (1-based)", min_value=1, max_value=max(1, total_rows), value=1, step=1)
            with col_b:
                end_idx = st.number_input("End row (inclusive, 1-based)", min_value=1, max_value=max(1, total_rows), value=min(total_rows, 20), step=1)

            # Normalize indices to 0-based half-open
            start0 = int(start_idx) - 1
            end0 = int(end_idx)

            if start0 >= end0:
                st.warning("Start row must be less than End row. Adjust the range to proceed.")
            else:
                run_batch = st.button("Run batch", key="run_batch")
                if run_batch:
                    rows_to_process = end0 - start0
                    results_rows = []
                    progress_bar = st.progress(0)
                    status = st.empty()
                    status.info(f"Processing {rows_to_process} comments...")

                    # Process items one by one to update progress
                    for i, comment in enumerate(df_preview["comment"].iloc[start0:end0].tolist(), start=1):
                        try:
                            label, score = run_sentiment_single(comment)
                        except Exception:
                            label, score = "ERROR", 0.0
                        try:
                            meaning = run_translate_single(comment)
                        except Exception:
                            meaning = ""
                        results_rows.append({
                            "original_comment": comment,
                            "sentiment_label": label,
                            "sentiment_confidence": score,
                            "meaning_in_english": meaning
                        })
                        progress_bar.progress(int(i / rows_to_process * 100))
                        # brief sleep to keep UI responsive for very fast local models (no effect on real latency)
                        time.sleep(0.05)

                    progress_bar.empty()
                    status.success("Batch processing complete.")
                    result_df = pd.DataFrame(results_rows)

                    st.markdown("#### Sample results")
                    st.dataframe(result_df.head(10), use_container_width=True)

                    # Prepare CSV for download
                    csv_buf = io.StringIO()
                    result_df.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")
                    st.download_button("Download results (CSV)", data=csv_bytes, file_name="analysis_results.csv", mime="text/csv")

        else:
            st.info("No batch file uploaded yet. Upload a CSV or TXT file to enable Batch Mode.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
