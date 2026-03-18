import streamlit as st
from transformers import pipeline, AutoTokenizer

def sentiment(user_input):
    sentiment_pipeline = pipeline(model="ivanwonghs/trial_1")
    sentiment_result = sentiment_pipeline(user_input)
    sentiment = sentiment_result[0]["label"]
    confidence = sentiment_result[0]["score"]

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")
        
def translate(user_input):
    translate_pipeline = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    # The model name is known from earlier cells
    model_name = "Qwen/Qwen3-0.6B"
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "user", "content": "Just give me '"+user_input+"' in English purely in string charater"},
    ]
     # Apply chat template with thinking disabled
    # tokenize=False is important here as translate_pipeline expects string input now, not tokenized IDs
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Disable thinking mode
    )
    # Call the pipeline with the pre-processed text and desired max_new_tokens
    # The 'translate_pipeline' object is assumed to be defined in a previous cell (drax9z03oF-i)
    outputs = translate_pipeline(text_input, max_new_tokens=32768)
    generated_text_full = outputs[0]['generated_text']
    # Define the marker after which the actual response starts
    marker_end_think = "</think>\n\n"
    # Find the index of the marker
    start_of_response_idx = generated_text_full.rfind(marker_end_think)
    # Extract the substring after the marker
    raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
    # The response is typically enclosed in double quotes, strip them
    extracted_response = raw_response.strip().strip('"')
    st.write(f"Meaning in English: {extracted_response}")

def main():
    st.title("Muti-language Comment Analyser\n")
    st.write("Please input the comment you want to analyse:")
    user_input = st.text_input("")
    if user_input:
        sentiment(user_input)
        translate(user_input)
        
if __name__ == "__main__":
    main()
