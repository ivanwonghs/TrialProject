import streamlit as st
from transformers import pipeline

def main():
    sentiment_pipeline = pipeline(model="ivanwonghs/trial_1")
    #replyMsg_pipeline = pipeline(model="microsoft/Phi-4-mini-instruct")

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("")
    if user_input:
        sentiment_result = sentiment_pipeline(user_input)
        sentiment = sentiment_result[0]["label"]
        confidence = sentiment_result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
        
        #replyMsg_result = replyMsg_pipeline(f"Generate a polite reply to apologize in corresponding language for below message: '{user_input}'")
        #st.write(replyMsg_result)
        #st.write(f"Reply Message: {confidence:.2f}")

if __name__ == "__main__":
    main()
