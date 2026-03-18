import streamlit as st
from transformers import pipeline

def main():
    sentiment_pipeline = pipeline(model="ivanwonghs/trial_1")
    replyMsg_pipeline = pipeline("text-generation", model="facebook/xglm-564M")

    st.title("Muti-language Reply Message Generator For Negative Comment")
    st.write("Enter a comment to generate reply message in the corresponding language: ")

    user_input = st.text_input("")
    if user_input:
        sentiment_result = sentiment_pipeline(user_input)
        sentiment = sentiment_result[0]["label"]
        confidence = sentiment_result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
        
        replyMsg_result = replyMsg_pipeline(f"Generate a polite reply to apologize in the same language language for below message: '{user_input}'")
        st.write(f"Suggested Reply Message: {replyMsg_result[0]['generated_text']}")

if __name__ == "__main__":
    main()
