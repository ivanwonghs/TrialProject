import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Tuple
import pandas as pd
import io
import time

# Page config
st.set_page_config(page_title="Multilingual Comment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# Internal (hard-coded) model IDs
_TRANSLATE_MODEL_ID = "Qwen/Qwen3-0.6B"
_SENTIMENT_MODEL_ID = "ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final_version"

# Cached pipeline stores
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

# Inference helpers
def run_sentiment_single(text: str) -> Tuple[str, float]:
    pipeline_obj = get_sentiment_pipeline_cached(_SENTIMENT_MODEL_ID)
    result = pipeline_obj(text)
    return result[0].get("label", "UNKNOWN"), result[0].get("score", 0.0)

def run_translate_single(text: str) -> str:
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

# File parsing helper
def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["comment"])
    name = uploaded_file.name.lower()
    try:
        uploaded_file.seek(0)
        if name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            df = pd.DataFrame({"comment": lines})
        else:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            text_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if text_cols:
                df = df[[text_cols[0]]].rename(columns={text_cols[0]: "comment"})
            else:
                df = df.rename(columns={df.columns[0]: "comment"}) if len(df.columns) > 0 else pd.DataFrame(columns=["comment"])
    except Exception:
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode("utf-8")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            df = pd.DataFrame({"comment": lines})
        except Exception:
            df = pd.DataFrame(columns=["comment"])
    df["comment"] = df["comment"].astype(str).fillna("").str.strip()
    df = df[df["comment"] != ""].reset_index(drop=True)
    return df

# Main UI
def main():
    # Title row (compact)
    title_col, meta_col = st.columns([0.8, 0.2])
    with title_col:
        st.title("Multilingual Social Media Product Comment Analyzer")
        st.caption("Quickly get sentiment and an English rendering for short comments.")
    with meta_col:
        st.markdown("")  # reserved

    # Page CSS for compact two-column layout
    st.markdown(
        """
        <style>
        .layout { display:flex; gap:20px; align-items:flex-start; padding-top:6px; }
        .left { width:340px; padding:12px; box-sizing:border-box; border-right:1px solid rgba(0,0,0,0.06); }
        .right { flex:1 1 auto; padding:12px; box-sizing:border-box; }
        .about-small { font-size:14px; line-height:1.4; color: #111827; }
        .muted { color:#6b7280; font-size:13px; }
        .langs { font-size:14px; line-height:1.6; }
        @media (max-width: 900px) {
            .layout { flex-direction:column; }
            .left { width:auto; border-right:none; border-bottom:1px solid rgba(0,0,0,0.06); }
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="layout">', unsafe_allow_html=True)
    left_col, right_col = st.columns([0.28, 0.72])

    # Left column: About + Supported languages
    with left_col:
        st.markdown('<div class="left">', unsafe_allow_html=True)
        st.subheader("About")
        st.markdown(
            """
            <div class="about-small">
            This app analyzes short social-media or product comments in many languages.
            - Click Analyze for a single comment.
            - Or use Batch Mode to upload a file and process many comments at once.
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("**How it works**")
        st.markdown(
            """
            1. Predicts sentiment (label + confidence).  
            2. Produces a concise English rendering for non-native readers.  
            3. Batch results are downloadable as CSV.
            """
        )
        st.markdown("**Batch mode quick tips**")
        st.markdown(
            """
            - CSV: first text column is used automatically.  
            - TXT: one comment per line.  
            - Preview results and download when done.
            """
        )
        st.markdown("**Supported languages**")
        st.markdown(
            """
            <div class="langs">
            - English<br>
            - Arabic<br>
            - German<br>
            - Spanish<br>
            - French<br>
            - Japanese<br>
            - Chinese<br>
            - Indonesian<br>
            - Hindi<br>
            - Italian<br>
            - Malay<br>
            - Portuguese
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Right column: Single comment + Batch
    with right_col:
        st.markdown('<div class="right">', unsafe_allow_html=True)

        # Single comment area
        st.subheader("Single comment")
        st.markdown("Enter a short comment and click Analyze.")
        # Changed placeholder to use '產品' instead of '服務'
        user_input = st.text_input("Comment to analyse", key="single_input", placeholder="e.g. 這個產品真的很差")

        btn_col1, btn_col2 = st.columns([0.5, 0.5])
        with btn_col1:
            analyze_clicked = st.button("Analyze", key="analyze_single")
        with btn_col2:
            demo_clicked = st.button("Run demo example", key="demo_btn")

        status_placeholder = st.empty()
        sentiment_placeholder = st.empty()
        translate_placeholder = st.empty()

        if demo_clicked:
            # Demo text changed from '服務' to '產品'
            demo_text = "這個產品真的很差"
            status_placeholder.info("Running demo example...")
            try:
                label, score = run_sentiment_single(demo_text)
                sentiment_placeholder.markdown(f"**Sentiment:** {label} ({score:.2%} confidence)")
            except Exception:
                sentiment_placeholder.error("Sentiment analysis failed. Please try again later.")
            try:
                meaning = run_translate_single(demo_text)
                translate_placeholder.markdown(f"**Meaning in English:** {meaning}")
            except Exception:
                translate_placeholder.error("Translation failed. Please try again later.")
            status_placeholder.success("Demo complete.")

        if analyze_clicked:
            if not user_input:
                st.warning("Please enter a comment to analyze.")
            else:
                with st.spinner("Analyzing comment — please wait..."):
                    status_placeholder.info("Running analysis...")
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

        # Batch mode area (unchanged)
        st.subheader("Batch mode")
        st.markdown("Upload a CSV (one text column) or a TXT file (one comment per line). Preview, select rows, then Run batch.")

        uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"], accept_multiple_files=False, key="batch_upload")

        df_preview = pd.DataFrame(columns=["comment"])
        if uploaded_file is not None:
            df_preview = parse_uploaded_file(uploaded_file)

        if not df_preview.empty:
            st.markdown("Preview (first 10)")
            st.dataframe(df_preview.head(10), use_container_width=True)
            total_rows = len(df_preview)
            st.markdown(f"Detected {total_rows} comments.")

            col1, col2 = st.columns(2)
            with col1:
                start_idx = st.number_input("Start row (1-based)", min_value=1, max_value=total_rows, value=1, step=1)
            with col2:
                end_idx = st.number_input("End row (inclusive)", min_value=1, max_value=total_rows, value=min(20, total_rows), step=1)

            start0 = int(start_idx) - 1
            end0 = int(end_idx)

            if start0 >= end0:
                st.warning("Start row must be less than End row.")
            else:
                if st.button("Run batch", key="run_batch"):
                    rows_to_process = end0 - start0
                    results_rows = []
                    progress_bar = st.progress(0)
                    status = st.empty()
                    status.info(f"Processing {rows_to_process} comments...")

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
                        time.sleep(0.02)

                    progress_bar.empty()
                    status.success("Batch processing complete.")
                    result_df = pd.DataFrame(results_rows)
                    st.markdown("Sample results")
                    st.dataframe(result_df.head(10), use_container_width=True)

                    csv_buf = io.StringIO()
                    result_df.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode("utf-8")
                    st.download_button("Download results (CSV)", data=csv_bytes, file_name="analysis_results.csv", mime="text/csv")
        else:
            st.info("No batch file uploaded yet. Upload a CSV or TXT to enable Batch Mode.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
