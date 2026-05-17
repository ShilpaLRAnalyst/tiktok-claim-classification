import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. SET UP THE WEBPAGE LAYOUT & TITLES
st.set_page_config(page_title="TikTok Claim Detector", page_icon="📊", layout="centered")
st.title("📊 TikTok Video Claim Classification")
st.markdown("### Champion Model: Random Forest Classifier")
st.write("This app uses a trained machine learning model to predict whether a TikTok video contains a **Claim** or an **Opinion** based on its engagement metrics.")

# 2. LOAD THE SERIALIZED MODEL AND VECTORIZER
@st.cache_resource
def load_assets():
    # Load the Random Forest model brain
    with open("tiktok_random_forest_backup.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Load the text tokenizer asset
    with open("tiktok_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
        
    return model, vectorizer

model, vectorizer = load_assets()

# 3. CREATE THE USER INPUT INTERFACE
st.subheader("📁 Enter Video Engagement Metrics")

video_duration_sec = st.slider("Video Duration (seconds)", min_value=5, max_value=60, value=30)
video_view_count = st.number_input("Video View Count", min_value=0, value=150000, step=5000)
video_like_count = st.number_input("Video Like Count", min_value=0, value=5000, step=500)
video_share_count = st.number_input("Video Share Count", min_value=0, value=500, step=50)
video_download_count = st.number_input("Video Download Count", min_value=0, value=200, step=20)
video_comment_count = st.number_input("Video Comment Count", min_value=0, value=100, step=10)

# Direct text entry box for token simulation
user_transcription = st.text_input("Video Transcription Text Preview", value="This video contains factual information about trending policy changes.")
text_length = len(user_transcription)

# 4. PREDICTION PROCESSING
st.markdown("---")
if st.button("🚀 Run Claim Analysis", type="primary"):
    
    # A. Build the base 10 features dataframe (no raw text column, matching your exact order!)
    base_features = pd.DataFrame([{
        'video_duration_sec': video_duration_sec,
        'video_view_count': video_view_count,
        'video_like_count': video_like_count,
        'video_share_count': video_share_count,
        'video_download_count': video_download_count,
        'video_comment_count': video_comment_count,
        'text_length': text_length,
        'verified_status_verified': 0,  
        'author_ban_status_banned': 0,
        'author_ban_status_under review': 0
    }])
    
    # B. Vectorize the raw text into the 15 token columns using the loaded vectorizer config
    text_token_matrix = vectorizer.transform([user_transcription]).toarray()
    
    # Convert token counts to a matching DataFrame structure with alphabetical token columns
    token_features = pd.DataFrame(text_token_matrix, columns=vectorizer.get_feature_names_out())
    
    # C. Concatenate them horizontally to form the exact 25-column dataset your model expects!
    input_features = pd.concat([base_features, token_features], axis=1)
    
    # Run the model prediction
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)
    
    # 5. DISPLAY THE RESULTS VISUALLY
    st.subheader("🔮 Model Analysis Result:")
    
    if prediction[0] == 1:
        st.error(f"🚨 **Prediction: CLAIM**")
        st.write(f"The model is highly confident ({prediction_proba[0][1]*100:.1f}%) that this video presents a factual claim that requires human verification.")
    else:
        st.success(f"💬 **Prediction: OPINION**")
        st.write(f"The model predicts with {prediction_proba[0][0]*100:.1f}% confidence that this video represents a personal opinion.")