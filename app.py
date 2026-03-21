import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Optional

st.set_page_config(page_title="Multilingual Comment Analyzer", layout="wide")

# Lazy global cache for pipelines/tokenizer so we don't reload on every call
_SENTIMENT_PIPELINE: Optional[object] = None
_TRANSLATE_PIPELINE: Optional[object] = None
_TRANSLATE_TOKENIZER: Optional[object] = None

@st.cache_resource
def get_sentiment_pipeline():
    # Use a cached resource so the model loads only once per session/app lifetime
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(model="ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final")
    return _SENTIMENT_PIPELINE

@st.cache_resource
def get_translate_pipeline_and_tokenizer():
    global _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER
    model_name = "Qwen/Qwen3-0.6B"
    if _TRANSLATE_PIPELINE is None:
        _TRANSLATE_PIPELINE = pipeline("text-generation", model=model_name)
    if _TRANSLATE_TOKENIZER is None:
        _TRANSLATE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER

def sentiment(user_input: str, placeholder):
    # placeholder is an st.empty() where results will be written
    pipeline_obj = get_sentiment_pipeline()
    # Run inference (this is where we want the spinner/placeholder to show loading)
    sentiment_result = pipeline_obj(user_input)
    sentiment_label = sentiment_result[0]["label"]
    confidence = sentiment_result[0]["score"]

    # Replace placeholder content with the result
    placeholder.markdown(f"**Sentiment:** {sentiment_label} ({confidence:.2%} confidence)")

def translate(user_input: str, placeholder):
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer()

    # Build messages and apply chat template as you had
    messages = [
        {"role": "user", "content": "Translate the following into English: '"+user_input+"' "},
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    outputs = translate_pipeline(text_input, max_new_tokens=32768)
    generated_text_full = outputs[0].get('generated_text', "")

    marker_end_think = "</think>\n\n"
    start_of_response_idx = generated_text_full.rfind(marker_end_think)
    if start_of_response_idx != -1:
        raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
    else:
        # fallback if marker not found, take whole generated text
        raw_response = generated_text_full

    extracted_response = raw_response.strip().strip('"')
    placeholder.markdown(f"**Meaning in English:** {extracted_response}")

def main():
    st.markdown("## Multilingual Social Media Product Comment Analyzer\n")
    st.divider()

    # Layout: left column for language/support info, right column for app input/results
    left_col, divider_col, right_col = st.columns([1, 0.02, 3])

    with left_col:
        # Place multilingual support markdown block on the left
        st.markdown(
            """

                Supported languages:
                - English
                - Arabic
                - German
                - Spanish
                - French
                - Japanese
                - Chinese
                - Indonesian
                - Hindi
                - Italian
                - Malay
                - Portuguese
            
            """
        )

    with divider_col:
        # Add a vertical line using a small column and HTML/CSS
        st.markdown(
            """
            <div style="height:100%; display:flex; align-items:stretch;">
                <div style="width:1px; background-color:rgba(0,0,0,0.12); margin-left:auto; margin-right:auto;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with right_col:
        user_input = st.text_input("Please input the comment you want to analyse:")

        if user_input:
            # Create placeholders for results
            status_placeholder = st.empty()      # for overall status / loading screen text
            sentiment_placeholder = st.empty()   # will hold sentiment result
            translate_placeholder = st.empty()   # will hold translation result

            # Show a loading screen/message while work runs
            with st.spinner("Analyzing comment — this may take a while..."):
                status_placeholder.info("Loading models and running inference. Please wait...")

                # Run sentiment -> update sentiment_placeholder when done
                sentiment(user_input, sentiment_placeholder)

                # Run translation -> update translate_placeholder when done
                translate(user_input, translate_placeholder)

                # Once done, remove the status/loading message
                status_placeholder.success("Analysis complete.")

if __name__ == "__main__":
    main()

    #TEST
    st.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    st.write(pipeline(model="ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final"))
    st.write(pipeline(model="ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final")("TTESTING"))
