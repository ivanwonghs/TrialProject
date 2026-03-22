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
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(model="ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final")
    return _SENTIMENT_PIPELINE

@st.cache_resource
def get_translate_pipeline_and_tokenizer(model_name="Qwen/Qwen3-0.6B"):
    global _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER
    if _TRANSLATE_PIPELINE is None:
        _TRANSLATE_PIPELINE = pipeline("text-generation", model=model_name)
    if _TRANSLATE_TOKENIZER is None:
        _TRANSLATE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER

def sentiment(user_input: str, placeholder):
    pipeline_obj = get_sentiment_pipeline()
    sentiment_result = pipeline_obj(user_input)
    sentiment_label = sentiment_result[0]["label"]
    confidence = sentiment_result[0]["score"]
    placeholder.markdown(f"**Sentiment:** {sentiment_label} ({confidence:.2%} confidence)")

def translate(user_input: str, placeholder, logic_template: str = None, model_name: str = "Qwen/Qwen3-0.6B"):
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer(model_name=model_name)

    if logic_template and logic_template.strip():
        # If a custom logic/prompt template provided, use it; substitute {text} if present
        if "{text}" in logic_template:
            content_to_translate = logic_template.replace("{text}", user_input)
        else:
            # Otherwise append the user text to the template
            content_to_translate = logic_template.strip() + "\n\nText: " + user_input
    else:
        content_to_translate = "Translate the following into English: '" + user_input + "'"

    messages = [
        {"role": "user", "content": content_to_translate},
    ]

    # Use tokenizer's chat template if available; otherwise fall back to raw content
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        # Some tokenizers/models might not implement apply_chat_template
        text_input = content_to_translate

    outputs = translate_pipeline(text_input, max_new_tokens=1024)
    generated_text_full = outputs[0].get('generated_text', "")

    marker_end_think = "</think>\n\n"
    start_of_response_idx = generated_text_full.rfind(marker_end_think)
    if start_of_response_idx != -1:
        raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
    else:
        raw_response = generated_text_full

    extracted_response = raw_response.strip().strip('"')
    placeholder.markdown(f"**Meaning in English:** {extracted_response}")

def main():
    st.markdown("## Multilingual Social Media Product Comment Analyzer\n")
    st.markdown(
        """
        <style>
        /* Make the two columns and the vertical divider full height of content */
        .split-row {
            display: flex;
            gap: 16px;
            align-items: stretch;
        }
        .left-pane {
            flex: 1 1 320px;
            max-width: 420px;
        }
        .right-pane {
            flex: 3 1 800px;
        }
        .vertical-divider {
            width: 1px;
            background-color: rgba(0,0,0,0.12);
            margin: 0 8px;
        }
        /* ensure inner Streamlit elements don't collapse the container */
        .stContainer {
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create manual split using HTML/CSS wrappers so divider is full-height
    container = st.container()
    with container:
        cols_html = """
        <div class="split-row">
            <div class="left-pane" id="left-pane"></div>
            <div class="vertical-divider"></div>
            <div class="right-pane" id="right-pane"></div>
        </div>
        """
        st.markdown(cols_html, unsafe_allow_html=True)

        # To place Streamlit elements into the left and right 'logical' columns,
        # we'll create two columns in proportion matching the CSS above and render into them.
        left_col, mid_col, right_col = st.columns([1, 0.02, 3])

        with left_col:
            st.markdown("### Controls")
            # MODEL selection (you can add other model names if desired)
            model_choice = st.selectbox(
                "Model (translation pipeline)",
                options=[
                    "Qwen/Qwen3-0.6B",
                    # add other model ids here if you want
                ],
                index=0
            )
            # FUNCTION selection: Sentiment, Translate, Both
            function_choice = st.selectbox(
                "Function",
                options=["Both", "Sentiment", "Translate"],
                index=0
            )

            # Show supported languages list in left pane too
            st.markdown("#### Supported languages")
            st.markdown(
                """
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

                    # Decide what to run based on Function selection
                    if function_choice in ("Both", "Sentiment"):
                        try:
                            sentiment(user_input, sentiment_placeholder)
                        except Exception as e:
                            sentiment_placeholder.error(f"Sentiment error: {e}")

                    if function_choice in ("Both", "Translate"):
                        try:
                            translate(user_input, translate_placeholder, logic_template=logic_template, model_name=model_choice)
                        except Exception as e:
                            translate_placeholder.error(f"Translate error: {e}")

                    # Once done, remove the status/loading message
                    status_placeholder.success("Analysis complete.")

if __name__ == "__main__":
    main()
