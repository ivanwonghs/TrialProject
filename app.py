import streamlit as st
from transformers import pipeline, AutoTokenizer

def sentiment():
    sentiment_pipeline = pipeline(model="ivanwonghs/trial_1")
    sentiment_result = sentiment_pipeline(user_input)
    sentiment = sentiment_result[0]["label"]
    confidence = sentiment_result[0]["score"]

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")
    
    replyMsg_result = replyMsg_pipeline(f"Generate a polite reply to apologize in the same language language for below message: '{user_input}'")
    st.write(f"Suggested Reply Message: {replyMsg_result[0]['generated_text']}")
        
def translate():
    # The model name is known from earlier cells
    model_name = "Qwen/Qwen3-0.6B"
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "user", "content": "Just give me 'あたしの夫日本人だから毎日日本語勉強します。日本語ちゃんと話すと彼の嬉し顔見ますのは楽しみです。やくにたつ全部ビデオありがとう' in English purely in string charater"},
    ]
     # Apply chat template with thinking disabled
    # tokenize=False is important here as pipe expects string input now, not tokenized IDs
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Disable thinking mode
    )
    # Call the pipeline with the pre-processed text and desired max_new_tokens
    # The 'pipe' object is assumed to be defined in a previous cell (drax9z03oF-i)
    outputs = pipe(text_input, max_new_tokens=32768)
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
        sentiment()
        translate()
        
if __name__ == "__main__":
    main()
