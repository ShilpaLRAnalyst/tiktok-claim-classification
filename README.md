# 🎬 TikTok Video Claim Classification Web Application

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline built to classify TikTok video content into either a factual **Claim** or a personal **Opinion**. This project combines text feature extraction with social engagement metrics to serve live predictions via an interactive web interface.

🔗 **[Live Streamlit App Link](https://tiktok-claim-classification-hvqqvchfq9p48fqzetx4z8.streamlit.app/)**  | **[Jupyter Notebook Pipeline](https://nbviewer.org/github/ShilpaLRAnalyst/tiktok-claim-classification/blob/main/TikTok_Claim_Classification_Project.ipynb)** | **[Google Colab Mirror](https://colab.research.google.com/drive/1uOgK5D-yhrL7ZCv04Yn_4WlnkidRJQhH?usp=sharing)**

---

## 🚀 Key Features & Pipeline Architecture
* **Unstructured Text Processing (NLP):** Utilizes a pre-trained `CountVectorizer` to tokenize, clean, and vectorize raw video transcripts for model training.
* **Feature Engineering Pipeline:** Combines live, user-input engagement metrics (views, likes, shares, comments) with NLP features using a scaled 25-column alignment matrix.
* **Predictive Performance:** Powered by a tuned **Random Forest Classifier**. While XGBoost achieved strong overall accuracy, it yielded a higher rate of **false negatives**. Since content safety requires minimizing missed claims, Random Forest was selected for production due to its superior recall and ability to reliably capture true claims.
* **Cloud Architecture:** Packaged with standalone Python ecosystem dependencies and deployed via Streamlit Community Cloud and GitHub version control workflows.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Machine Learning & NLP:** Scikit-Learn,XGBoost,Pandas, NumPy, Pickle
* **Deployment & UI Framework:** Streamlit Framework, GitHub Webhooks, Cloud Infrastructure Server

---

## 📁 Repository Contents
* `TikTok_Claim_Classification_Project.ipynb`: The master engineering notebook containing exploratory data analysis, data cleaning, and feature encoding. It showcases the comparative evaluation of **Logistic Regression**, **XGBoost**, and **Random Forest** models, detailing hyperparameter tuning and champion model selection metrics.
* `app.py`: The master execution script hosting the user interface logic, database structural layout, and cloud prediction pipeline.
* `requirements.txt`: Lists the required Python packages and dependencies needed to run and host the application.
* `tiktok_random_forest_backup.pkl`: The serialized Random Forest model trained and used for production predictions.
* `tiktok_vectorizer.pkl`: The serialized CountVectorizer instance used to transform raw text inputs.

---

## 💡 Industry Application
Automated claim detection tools are critical components in modern social media ecosystems, helping content management teams scale moderation workflows, filter out misinformative claims, and segment opinion-based editorial content efficiently.
