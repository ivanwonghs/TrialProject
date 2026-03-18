import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Optional

# Lazy global cache for pipelines/tokenizer so we don't reload on every call
_SENTIMENT_PIPELINE: Optional[object] = None
_TRANSLATE_PIPELINE: Optional[object] = None
_TRANSLATE_TOKENIZER: Optional[object] = None

def get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        # change model name if needed
        _SENTIMENT_PIPELINE = pipeline(model="ivanwonghs/trial_1")
    return _SENTIMENT_PIPELINE

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
    placeholder.markdown(f"**Sentiment:** {sentiment_label}\n\n**Confidence:** {confidence:.2f}")

def translate(user_input: str, placeholder):
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer()

    # Build messages and apply chat template as you had
    messages = [
        {"role": "user", "content": "Just give me '"+user_input+"' in English purely in string charater"},
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
    st.title("Multi-language Comment Analyser")
    st.write("Please input the comment you want to analyse:")

    user_input = st.text_input("Enter comment here")

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

        # Optional: collapse the spinner by doing nothing else; results are shown in placeholders

if __name__ == "__main__":
    main()
